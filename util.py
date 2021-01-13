import re


def GetTokenIndices(passage, spans, tokenizer, overall_offset=0, max_length=-1):
  if not spans:
    # This would happen anyways, but saves iterating through the tokens
    return tokenizer.encode(passage, add_special_tokens=False), spans
  for start, end in spans:
    passage = passage[:start] + tokenizer.sep_token + passage[start:end] + tokenizer.sep_token + passage[end:]

  context_tokens = tokenizer.encode(passage, add_special_tokens=False)

  # Remove added sep tokens to get link indices
  token_spans = []
  offset = 0
  span_start = -1
  length = min(len(context_tokens), max_length) if max_length > 0 else len(context_tokens)
  for i in range(length):
    token = context_tokens[i]
    if token == tokenizer.sep_token_id:
      if span_start > -1:
        offset += 1
        token_spans.append((span_start, i-offset + overall_offset))
        span_start = -1
      else:
        span_start = i-offset + overall_offset
        offset += 1

        
  for start, end in token_spans:
    start -= overall_offset
    end -= overall_offset
    assert context_tokens.pop(start) == tokenizer.sep_token_id
    assert context_tokens.pop(end+1) == tokenizer.sep_token_id
  
  return context_tokens, token_spans


def RetrieveArticle(title, context_articles): 
  article = None
  if title in context_articles:
    article = context_articles[title]
  elif title.lower() in context_articles:
    article = context_articles[title.lower()]
  if article is not None:
    # Remove html tags (mainly links)
    article = re.sub("<[^>]*>", "", article)
  return article
