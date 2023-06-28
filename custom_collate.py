max_length = 128

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
def torch_mask_tokens(inputs, mlm_probability = 0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    import torch

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

# result type float cannot be cast as long?
# samples is a list of dicts
def custom_collate_fn(samples):
    # extract samples from data, pad and aggregate them
    samples = [s['input_ids'] for s in samples]
    max_len = max([len(s) for s in samples])
    samples = [F.pad(s, pad=(0,max_len-len(s))) for s in samples]
    input_ids = torch.stack(samples)
    labels = torch.stack(samples)

    # Create an attention mask
    attention_mask = input_ids != 0

    # Create a mask tensor
    mask = F.one_hot(input_ids, num_classes=tokenizer.vocab_size)
    mask = mask.float()

    # Randomly mask some tokens with a given probability
    mask = F.dropout(mask, p=0.15)

    # Return the batch of data
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "mask": mask}
