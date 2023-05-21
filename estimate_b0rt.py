
# b0rt lives!
import pandas as pd
import torch
from transformers import pipeline, BertForMaskedLM

reload_bert = False

# reload the BERT model from package
if reload_bert==True:
    from transformers import BertModel, BertTokenizer, BertConfig
    model_name = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)

# this works with the BertForMaskLM! but not BertModel, hugging face is so annoying!
model_zeros = BertForMaskedLM.from_pretrained("bert_initial")

test_masks = ["the [MASK] in the hat",
"the lion the [MASK] and the wardrobe",
"[MASK] and prejudice",
"one [MASK] years of solitude",
"brave [MASK] world",
"the count of [MASK] cristo",
"in [MASK] of lost time",
"[MASK] and peace",
"one [MASK] over the cuckoo's nest",
"the [MASK] of the rings",
"the [MASK] in the rye",
"alice's [MASK] in wonderland",
"charlie and the [MASK] factory",
"a tale of [MASK] cities",
"the [MASK] of war"]

def model_tests(zero_frac):

    model_zeros = BertForMaskedLM.from_pretrained("bert_initial")

    with torch.no_grad(): 
        for i, (name, weight) in enumerate(model_zeros.named_parameters()):
            if weight.shape==torch.Size([768,768]):
                weight.data = weight.data * torch.bernoulli(torch.ones(weight.shape),zero_frac)

    model_zeros.save_pretrained("b0rt")

    plz = pipeline('fill-mask', model='b0rt')
    top_results = []

    for phrase in test_masks:
        top_results.append(plz(phrase)[0])

    return pd.DataFrame(top_results)


# evaluate b0rt performance on masked benchmarks
# try zeroing out just specific layers or blocks

