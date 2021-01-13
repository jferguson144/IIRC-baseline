import json
import torch
import sys
import argparse
import context_selection
import link_prediction
import os
import logging
import random
import util
from transformers import BertTokenizer, RobertaTokenizer

random.seed(1)

def main():
  logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
  with open(args.context_path) as in_file:
    context_articles = json.load(in_file)
  with open(args.train_path) as in_file:
    train_data = json.load(in_file)
  with open(args.dev_path) as in_file:
    dev_data = json.load(in_file)

  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  # Numnet uses roberta, so when adding sep tokens for numnet 
  #  input, we need to use roberta sep tokens
  roberta_tokenizer = RobertaTokenizer.from_pretrained("numnet_plus/drop_dataset/roberta.base")

  # If model is None, gold information is used
  if args.use_gold_links:   
    link_predictor = None
  else:
    link_predictor = link_prediction.LinkPredictor()
    if args.device == "cuda":
      link_predictor = link_predictor.cuda()
    link_predictor.load_state_dict(torch.load(args.link_path))
  
  if args.use_gold_context:
    retrieval_model = None
  else:
    retrieval_model = context_selection.PassageScoringModel()
    if args.device == "cuda":
      retrieval_model = retrieval_model.cuda()
    retrieval_model.load_state_dict(torch.load(args.retrieval_path))

  
  formatted_train = MakeDropStyle(train_data, context_articles, tokenizer, roberta_tokenizer,
                                  link_predictor=None, retrieval_model=None)
  with open(args.train_out_path, "w") as out_file:
    json.dump(formatted_train, out_file, indent=2)
  
    
  formatted_dev = MakeDropStyle(dev_data, context_articles, tokenizer, roberta_tokenizer, 
                                link_predictor, retrieval_model, 
                                initial_qid=len(formatted_train))
  with open(args.dev_out_path, "w") as out_file:
    json.dump(formatted_dev, out_file, indent=2)
  



def MakeDropStyle(data, context_articles, tokenizer, roberta_tokenizer,
                  link_predictor=None, retrieval_model=None, initial_qid=0):
  """
  Setting a model to None uses gold information for that step.
  """
  new_data = {}
  q_index = initial_qid
  with torch.no_grad():
    if retrieval_model is not None:
      length_mask = torch.zeros(args.batch_size, retrieval_model.inp_size, 
                                dtype=torch.int, device=args.device)
    for pid, passage in enumerate(data):
      if pid % 100 == 0:
        print("Working on %d / %d..." %(pid, len(data)))
      for question in passage["questions"]:
        q_index += 1
        # If no retrieval model, assume gold links
        if link_predictor is None or retrieval_model is None:
          # Grab gold context passages
          links = set([span["passage"] for span in question["context"] if span["passage"] != "main"])
        else:
          # Use link predictor to grab articles
          model_input, link_targets = link_prediction.MakeInput(passage["text"], question, 
                                                                passage["links"], tokenizer, device=args.device)
          link_scores = link_predictor(model_input)
          links = []
          for i in range(link_scores.shape[0]):
            if link_scores[i] > args.link_threshold:
              links.append(link_targets[i])

        if retrieval_model is None:
          if args.no_window:
            # Use the annotated context spans as is, putting main context first
            all_context = ([span["text"] for span in question["context"] if span["passage"] == "main"] + 
                           [span["text"] for span in question["context"] if span["passage"] != "main"])
          else:
            # Grab some surrounding context for each span
            all_context = []

            # Reduce the window size in the case of many links so all contexts can fit in the input
            window_size = min(args.max_context_size // (len(links) + 1), args.window_size)
            # Scale the stride according to the new size
            window_stride = int(args.window_stride / args.window_size * window_size)

            # Get main context
            gold_spans = [span for span in question["context"] if span["passage"] == "main"]
            for span in gold_spans:
              splits, labels, _ = context_selection.SplitArticle(passage["text"], [span], tokenizer, 
                                                              window_size, window_stride)
              gold_splits = [i for (i, l) in enumerate(labels) if l == 1]
              if gold_splits:
                all_context.append(tokenizer.decode(splits[random.choice(gold_splits)]))

            # Get linked contexts
            for link in links:
              gold_spans = [span for span in question["context"] if span["passage"] == link]
              article = util.RetrieveArticle(link, context_articles)
              for span in gold_spans:
                splits, labels, _ = context_selection.SplitArticle(article, [span], tokenizer, 
                                                                window_size, window_stride)
                gold_splits = [i for (i, l) in enumerate(labels) if l == 1]
                if gold_splits:
                  all_context.append(tokenizer.decode(splits[random.choice(gold_splits)]))
          if question["answer"]["type"] == "none" and args.unanswerable_noise:
            links = question["question_links"]
            for target in links: 
              article = util.RetrieveArticle(target, context_articles)
              if article is None or article.strip() == "":
                continue
              splits, _, _ = context_selection.SplitArticle(article, [], tokenizer, 
                                                         args.window_size, args.window_stride)
              all_context.append(tokenizer.decode(splits[0]))

        else:
          all_context = []
          # Reduce the window size in the case of many links so all contexts can fit in the input
          window_size = min(args.max_context_size // (len(links) + 1), args.window_size)
          # Scale the stride according to the new size
          window_stride = int(args.window_stride / args.window_size * window_size)

          # Predict main context
          splits, _, lengths = context_selection.SplitArticle(passage["text"], [], tokenizer,
                                                              window_size, window_stride, device=args.device)
          model_input, added_length = context_selection.MakeInput(question["question"], splits, 
                                                                  tokenizer, context=None, device=args.device)
          
          all_scores = torch.zeros(model_input.shape[0], device=args.device)
          for batch_start in range(0, model_input.shape[0], args.batch_size):
            batch_input = model_input[batch_start:batch_start+args.batch_size]
            batch_lengths = lengths[batch_start:batch_start+args.batch_size]
            for i in range(len(batch_lengths)):
              length_mask[i, :batch_lengths[i]+added_length] = 1
            scores = retrieval_model(batch_input, length_mask[:batch_input.shape[0], :batch_input.shape[1]])
            # Reset for later use
            length_mask[:] = 0
            all_scores[batch_start:batch_start+scores.shape[0]] = scores.detach()
          pred_index = all_scores.max(dim=0)[1]

          pred_main_context = tokenizer.decode(splits[pred_index])[:lengths[pred_index]]
          all_context.append(pred_main_context)

          # Predict linked contexts
          for target in links:
            article = util.RetrieveArticle(target, context_articles)
            if article is None or article.strip() == "":
              continue
            splits, _, lengths = context_selection.SplitArticle(article, [], tokenizer, window_size, 
                                                                window_stride, device=args.device)
            model_input, added_length = context_selection.MakeInput(question["question"], splits, tokenizer, 
                                                                    context=pred_main_context, device=args.device)
            all_scores = torch.zeros(model_input.shape[0], device=args.device)
            for batch_start in range(0, model_input.shape[0], args.batch_size):
              batch_input = model_input[batch_start:batch_start+args.batch_size]
              batch_lengths = lengths[batch_start:batch_start+args.batch_size]
              for i in range(len(batch_lengths)):
                length_mask[i, :batch_lengths[i]+added_length] = 1

              scores = retrieval_model(batch_input, length_mask[:batch_input.shape[0], :batch_input.shape[1]])
              # Reset for later use
              length_mask[:] = 0
            
              all_scores[batch_start:batch_start+scores.shape[0]] = scores.detach()
            pred_index = all_scores.max(dim=0)[1]
            pred_target_context = splits[pred_index][:lengths[pred_index]]
            all_context.append(tokenizer.decode(pred_target_context))

        # Format the data into drop style
        answer_info = question["answer"]
        a_type = answer_info["type"]
        if a_type == "span":
          answer_spans = [a["text"] for a in answer_info["answer_spans"]]
          answer_num = ""
        elif a_type == "value":
          answer_spans = []
          answer_num = answer_info["answer_value"]
        elif a_type == "binary":
          question = question
          answer_spans = [answer_info["answer_value"]]
          answer_num = ""
        elif a_type == "none":
          answer_spans = []
          answer_num = ""
          if args.filter_no_answer:
            q_index -= 1
            continue
        elif a_type == "bad":
          q_index -= 1
          continue
        answer_dict = {"date": {"day":"", "month":"", "year":""},
                       "number": answer_num,
                       "spans": answer_spans}
        new_data["q_%d" %(q_index)] = {"passage": roberta_tokenizer.sep_token.join(all_context),
                                       "qa_pairs": [{"question": question["question"],
                                                     "answer": answer_dict,
                                                     "validated_answers": [answer_dict],
                                                     "query_id":"qid_%d" %(q_index)}]}

  return new_data

      
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate drop-style input for final QA step.")
  parser.add_argument("-use_gold_links", default=False, action="store_true",
                      help="Use gold annotations for the link prediction step.")

  parser.add_argument("-use_gold_context", default=False, action="store_true",
                      help="Use gold annotations for the context selection step.")

  parser.add_argument("--window_size", type=int, default=100,
                      help="Sliding window size for context selection.")

  parser.add_argument("--window_stride", type=int, default=25,
                      help="Number of tokens to progress the sliding window for each step.")

  parser.add_argument("--train_path", type=str, default="iirc/train.json",
                      help="Path to IIRC training data.")

  parser.add_argument("--dev_path", type=str, default="iirc/dev.json",
                      help="Path to IIRC dev data.")

  parser.add_argument("--context_path", type=str, default="iirc/context_articles.json",
                      help="Path to IIRC context articles.")

  parser.add_argument("--link_path", type=str, default="models/link_predictor.model",
                      help="Path to link prediction model.")

  parser.add_argument("--retrieval_path", type=str, default="models/context_selector.model",
                      help="Path to context selection model.")

  parser.add_argument("--link_threshold", type=float, default=0.5,
                      help="Threshold score for links to be selected. In the range [0,1].")

  parser.add_argument("-no_window", default=False, action="store_true",
                      help="If using oracle spans, setting this does not use any additional surrounding context.")

  parser.add_argument("-filter_no_answer", default=False, action="store_true",
                      help="Remove unanswerable questions.")

  parser.add_argument("--train_out_path", type=str, default="iirc/drop_train.json",
                      help="Path at which to write drop-formatted training data.")

  parser.add_argument("--dev_out_path", type=str, default="iirc/drop_dev.json",
                      help="Path at which to write drop-formatted dev data.")

  parser.add_argument("-unanswerable_noise", default=False, action="store_true",
                      help="Use the first window from relevant links as context for unanswerable questions \
                      to prevent models from learning the shortcut that no context === unanswerable.")

  parser.add_argument("--max_context_size", type=int, default=450, 
                      help="Maximum number of context tokens allowed per example. Generally dependent on \
                      what downstream model is being used.")

  parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                      help="Whether to use cpu or gpu.")

  parser.add_argument("--batch_size", type=int, default=10,
                      help="Number of contexts to process simultaneously.")


  args = parser.parse_args()

  main()
