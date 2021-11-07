import torch


def vectorize(text,
              bert_model, bert_tokenizer,
              split_mode=True, bert_layers=4):
    """
    Vectorize text with BERT
    text - input text
    split_mode - use [SEP] token or not
    bert_layers - how much last layers concatenate
    """
    # Split sentences with [SEP] token
    if split_mode:
        text = text.strip().replace('. ', " [SEP] ")
    else:
        text = text.strip()
    # Add the special tokens from start and in the end
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = bert_tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

    if split_mode:
        segments_ids = [1 if word != '[SEP]' else 0
                        for word in tokenized_text[:-1]]
        segments_ids += [1]
    else:
        segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 24 layers.
    with torch.no_grad():
        try:
            outputs = bert_model(tokens_tensor, segments_tensors)
        except:
            return None
        hidden_states = outputs[2]

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states)

        # Current dimensions:
        # [# layers, # batches, # tokens, # features]
        # Desired dimensions:
        #[# tokens, # layers, # features]
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # Concatenate the last four layers.
        # Each vector will have length 4 x 1024 = 4,096
        # Stores the token vectors, with shape [22 x 4,096]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.
        # For each token in the sentence...
        for i in range(bert_layers):
            token_vecs = hidden_states[-2 - i][0]

            # Calculate the average of all token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
            token_vecs_cat.extend(sentence_embedding.tolist())
        return token_vecs_cat
