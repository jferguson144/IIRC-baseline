import json
import torch
import logging
import argparse
from transformers import BertTokenizer, BertModel
import os
import util

debug_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def main():
  logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
  with open(args.train_path) as in_file:
    train_questions = json.load(in_file)

  link_predictor = LinkPredictor()
  if args.device == "cuda":
    link_predictor = link_predictor.cuda()

  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  loss_function = torch.nn.BCELoss()
  optim = torch.optim.Adam(link_predictor.parameters(), lr=1e-5)
  optim.zero_grad()
  batch_count = 0
  for epoch in range(args.n_epochs):
    total_loss = 0
    print("Starting epoch %d..." %(epoch))
    for passage in train_questions:
      main_context = passage["text"]
      questions = passage["questions"]
      links = passage["links"]
      for question in questions:
        model_input, link_targets = MakeInput(main_context, question, links, tokenizer)
        link_scores = link_predictor(model_input)
        gold_labels = [1 if target in question["question_links"] else 0 for target in link_targets]
        label_tensor = torch.tensor(gold_labels, dtype=torch.float, device=args.device)
        loss = loss_function(link_scores, label_tensor)
        total_loss += loss.detach()
        loss.backward()
        batch_count += 1
        if batch_count >= args.update_size:
          optim.step()
          optim.zero_grad()
          batch_count = 0
    print("Loss at epoch %d: %.4f" %(epoch, total_loss))
    torch.save(link_predictor.state_dict(), os.path.join(args.model_dir, "%s-%d.model" %(args.model_name, epoch)))

def main_test():
  logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
  with open(args.train_path) as in_file:
    dev_questions = json.load(in_file)

  link_predictor = LinkPredictor()
  if args.device == "cuda":
    link_predictor = link_predictor.cuda()
  link_predictor.load_state_dict(torch.load("models/link_predictor.model"))

  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

  for passage in dev_questions:
    main_context = passage["text"]
    questions = passage["questions"]
    links = passage["links"]
    for question in questions:
      print(question["question"])
      model_input, link_targets = MakeInput(main_context, question, links, tokenizer)
      link_scores = link_predictor(model_input)
      gold_labels = [1 if target in question["question_links"] else 0 for target in link_targets]
      sorted_scores, sorted_indices = link_scores.sort(descending=True)
      print("Predictions:")
      for i in range(5):
        print(link_targets[sorted_indices[i]], ":", sorted_scores[i], "(", gold_labels[sorted_indices[i]], ")")
      print("-"*5)
      input()


def MakeInput(passage, question, links, tokenizer, device="cuda"):
  # Mark links using sep tokens, starting at the end (in util.GetTokenIndices)
  raw_link_indices = [x["indices"] for x in links][::-1]
  link_targets = [x["target"] for x in links][::-1]

  question_text = question["question"]
  question_tokens = tokenizer.encode(question_text)

  # Get token indices for links in context (offset by question tokens due to later concatenation)
  offset = len(question_tokens)
  context_tokens, link_spans = util.GetTokenIndices(passage, raw_link_indices, tokenizer, overall_offset=offset,
                                                    max_length=512-len(question_tokens) - 1)

  bert_tokens = question_tokens + context_tokens
  if len(bert_tokens) > 512:
    bert_tokens = bert_tokens[:512]
    
  # (1, length)
  bert_input = torch.tensor([bert_tokens], dtype=torch.long, device=device)
  sequence_ids = torch.tensor([([0] * (len(question_tokens)+1)) + 
                               ([1] * (len(bert_tokens) - len(question_tokens) - 1))], 
                              dtype=torch.long, device=device)

  return (bert_input, sequence_ids, link_spans), link_targets[:len(link_spans)]


class LinkPredictor(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.bert_model = BertModel.from_pretrained("bert-base-uncased")
    emb_dim = self.bert_model.config.hidden_size
    self.link_classifier = torch.nn.Sequential(torch.nn.Linear(emb_dim*2, 1), 
                                               torch.nn.Sigmoid())

  def forward(self, model_input):
    bert_input, sequence_ids, link_spans = model_input
    
    encoded_text = self.bert_model(bert_input, token_type_ids=sequence_ids)[0]
    
    # Grab first and last tokens for each link, and concatenate
    all_link_reps = encoded_text[0, link_spans].reshape([len(link_spans), -1])

    # (num_links)
    link_scores = self.link_classifier(all_link_reps).squeeze(1)
    return link_scores


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate drop-style input for final QA step.")
  parser.add_argument("--train_path", type=str, default="iirc/train.json",
                      help="Path to IIRC training data.")

  parser.add_argument("--n_epochs", type=int, default=10,
                      help="Number of epochs for training.")

  parser.add_argument("--update_size", type=int, default=10,
                      help="Number of questions to process before taking a step.")

  parser.add_argument("--model_dir", type=str, default="models",
                      help="Directory to which models will be written.")

  parser.add_argument("--model_name", type=str, default="link_predictor",
                      help="Base name for saving models.")
  
  parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                      help="Whether to use cpu or gpu.")

  args = parser.parse_args()

  main_test()
