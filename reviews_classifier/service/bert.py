import torch
from reviews_classifier.service import device
import numpy as np

def vectorize(text,
              model, tokenizer,
              split_mode=True, bert_layers=8):
    """
    Vectorize text
    text - input text
    split_mode - use [SEP] token or not
    bert_layers - how much last layers concatenate
    """
    #global pbar
    # Split sentences with [SEP] token
    if split_mode:
        text = text.strip().replace('. ', " [SEP] ")
    else:
        text = text.strip()
    # Add the special tokens from start and in the end
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    tokenized_text = tokenized_text[:512]
    tokenized_text[-1] = '[SEP]'
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    if split_mode:
        segments_ids = [
                1 if word != '[SEP]' and word != '[PAD]'
                else 0
                for word in tokenized_text[:-1]
                ]
        segments_ids += [1]
    else:
        segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens],
                                 dtype=torch.int,
                                 device=torch.device(device))
    segments_tensors = torch.tensor([segments_ids],
                                    dtype=torch.int,
                                    device=torch.device(device))
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 24 layers.
    with torch.no_grad():
        try:
            outputs = model(tokens_tensor, segments_tensors)
        except:
            return None
        hidden_states = outputs[2]

        token_vecs_cat = torch.stack(hidden_states[:-1 - bert_layers:-1],
                                     dim=1)[0]
        token_vecs_cat = torch.mean(token_vecs_cat, dim=1)
        token_vecs_cat = torch.reshape(token_vecs_cat, (bert_layers * 1024,))
        token_vecs_cat = token_vecs_cat.cpu().detach()\
                         .numpy().astype(np.float32)

        return token_vecs_cat
