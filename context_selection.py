import json
import torch
import argparse
import os
import util
from transformers import BertTokenizer, BertModel
import logging

def main():
  logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
  with open(args.context_path) as in_file:
    context_articles = json.load(in_file)
  with open(args.train_path) as in_file:
    data = json.load(in_file)

  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  model = PassageScoringModel()
  if args.device == "cuda":
    model = model.cuda()
  loss_function = torch.nn.BCELoss()
  optim = torch.optim.Adam(model.parameters(), lr=1e-5)
  
  batch_count = 0
  length_mask = torch.zeros(args.batch_size, model.inp_size, dtype=torch.int, device=args.device)
  for epoch in range(args.n_epochs):
    print("Starting epoch %d..." %(epoch))
    total_loss = 0
    count = 0
    for pid, passage in enumerate(data):
      if pid % 100 == 0:
        print("%d / %d" %(pid, len(data)))
      for question in passage["questions"]:
        count += 1
        context = question["context"]
        gold_main_spans = [span for span in context if span["passage"] == "main"]
        gold_main_text = [span["text"] for span in gold_main_spans]

        # Main context loss
        splits, labels, lengths = SplitArticle(passage["text"], gold_main_spans, tokenizer, 
                                               args.window_size, args.window_stride, device=args.device)
        model_input, added_length = MakeInput(question["question"], splits, tokenizer, context=None, device=args.device)
        for batch_start in range(0, model_input.shape[0], args.batch_size):
          batch_input = model_input[batch_start:batch_start + args.batch_size]
          label_batch = labels[batch_start:batch_start + args.batch_size]
          batch_lengths = lengths[batch_start:batch_start + args.batch_size]
          for i in range(len(batch_lengths)):
            length_mask[i, :batch_lengths[i] + added_length] = 1

          scores = model(batch_input, length_mask[:batch_input.shape[0], :batch_input.shape[1]])
          # Reset for later use
          length_mask[:] = 0
          loss = loss_function(scores, label_batch)
          total_loss += loss.detach()
          loss.backward()
          batch_count += 1

          if batch_count >= args.update_size:
            optim.step()
            optim.zero_grad()
            batch_count = 0

        # Retrieved article loss
        target_articles = set([span["passage"] for span in question["context"] if span["passage"] != "main"])        
        for title in target_articles:
          article = util.RetrieveArticle(title, context_articles)
          article_spans = [span for span in question["context"] if span["passage"] == title]
          splits, labels, lengths = SplitArticle(article, article_spans, tokenizer,
                                                 args.window_size, args.window_stride, device=args.device)
          model_input, added_length = MakeInput(question["question"], splits, tokenizer, context=gold_main_text, device=args.device)
          for batch_start in range(0, model_input.shape[0], args.batch_size):
            batch_input = model_input[batch_start:batch_start + args.batch_size]
            label_batch = labels[batch_start:batch_start + args.batch_size]
            batch_lengths = lengths[batch_start:batch_start + args.batch_size]
            for i in range(len(batch_lengths)):
              length_mask[i, :batch_lengths[i] + added_length] = 1


            scores = model(batch_input, length_mask[:batch_input.shape[0], :batch_input.shape[1]])
            # Reset for later use
            length_mask[:] = 0
            loss = loss_function(scores, label_batch)
            total_loss += loss.detach()
            loss.backward()
            batch_count += 1

            if batch_count >= args.batch_size:
              optim.step()
              optim.zero_grad()
              batch_count = 0
    print("Total loss at epoch %d: %.4f" %(epoch, total_loss))
    torch.save(model.state_dict(), os.path.join(args.model_dir, "%s-%d.model" %(args.model_name, epoch)))
  

def SplitArticle(article, context, tokenizer, window_size, window_stride, device="cuda"):
  """
  Providing context spans assigns labels
  """
  gold_spans = sorted([span["indices"] for span in context])[::-1]
  article_tokens, token_spans = util.GetTokenIndices(article, gold_spans, tokenizer)

  num_splits = len(article_tokens) // window_stride + (len(article_tokens) % window_stride != 0)
  splits = torch.ones(num_splits, window_size, 
                      dtype=torch.long, device=device) * tokenizer.pad_token_id
  labels = torch.zeros(num_splits, dtype=torch.float, device=device)
  split_index = 0
  lengths = []
  for window_start in range(0, len(article_tokens), window_stride):
    window_end = window_start + window_size
    split = article_tokens[window_start:window_end]
    splits[split_index, :len(split)] = torch.tensor(split, dtype=torch.long, device=device)
    lengths.append(len(split))
    for start, end in token_spans:
      if window_start <= start and window_end >= end:
        labels[split_index] = 1
    split_index += 1
  return splits, labels, lengths
  

def MakeInput(question, splits, tokenizer, context=None, device="cuda"):
  n_splits = splits.shape[0]
  sep_tokens = torch.ones(n_splits, 1,
                          dtype=torch.long, device=device)

  question_tensor = torch.tensor(tokenizer.encode(question), 
                                 dtype=torch.long, device=device)
  expanded_question = question_tensor.unsqueeze(0).expand(n_splits, -1)
  if context is not None:
    main_tensor = torch.tensor(tokenizer.encode(context, add_special_tokens=False), 
                               dtype=torch.long, device=device)
    expanded_main = main_tensor.unsqueeze(0).expand(n_splits, -1)
    # Questions are encoded with cls and sep, so no need to add additional sep there
    model_input = torch.cat([expanded_question, expanded_main, sep_tokens, splits], dim=1)
    added_length = expanded_question.shape[1] + expanded_main.shape[1] + 1
  else:
    model_input = torch.cat([expanded_question, splits], dim=1)
    added_length = expanded_question.shape[1]

  return model_input, added_length
  
  

class PassageScoringModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.bert_model = BertModel.from_pretrained("bert-base-uncased")
    self.emb_dim = self.bert_model.config.hidden_size
    self.inp_size = self.bert_model.config.max_position_embeddings
    self.passage_scorer = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, 1), 
                                               torch.nn.Sigmoid())

  def forward(self, tokens, length_mask):
    outputs = self.bert_model(tokens, attention_mask=length_mask)
    encoded_text = outputs[0]
    score = self.passage_scorer(encoded_text[:, 0]).squeeze(1)
    return score


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate drop-style input for final QA step.")
  parser.add_argument("--train_path", type=str, default="iirc/train.json",
                      help="Path to IIRC training data.")

  parser.add_argument("--context_path", type=str, default="iirc/context_articles.json",
                      help="Path to IIRC context articles.")

  parser.add_argument("--window_size", type=int, default=100,
                      help="Sliding window size for context selection.")

  parser.add_argument("--window_stride", type=int, default=25,
                      help="Number of tokens to progress the sliding window for each step.")

  parser.add_argument("--n_epochs", type=int, default=10,
                      help="Number of epochs for training.")

  parser.add_argument("--update_size", type=int, default=10,
                      help="Number of questions to process before taking a step.")

  parser.add_argument("--model_dir", type=str, default="models",
                      help="Directory to which models will be written.")

  parser.add_argument("--model_name", type=str, default="context_selector",
                      help="Base name for saving models.")

  parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                      help="Whether to use cpu or gpu.")

  parser.add_argument("--batch_size", type=int, default=10,
                      help="Number of contexts to process simultaneously.")
  
  args = parser.parse_args()

  main()
