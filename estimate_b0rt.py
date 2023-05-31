
# b0rt lives!
import pandas as pd
import torch
from transformers import pipeline, BertForMaskedLM

reload_bert = False

# reload the BERT model from package
if reload_bert==True:
    from transformers import BertTokenizer, BertConfig
    model_name = 'bert-base-uncased'
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    model.save_pretrained("b0rt")
    tokenizer.save_pretrained("b0rt")
    config.save_pretrained("b0rt")

test_answers = ["witch","pride","kill","wrath","young","hundred","monte","search",
                "war","punishment","rises","madame","heights","being","streetcar",
                "glass","rex","much","nest","lord","catcher","adventures",
                "chocolate","cities"]

test_masks = ["the lion the [MASK] and the wardrobe.",
"[MASK] and prejudice.",
"to [MASK] a mockingbird.",
"the grapes of [MASK].",
"a portrait of an artist as a [MASK] man.",
"one [MASK] years of solitude.",
"the count of [MASK] cristo.",
"in [MASK] of lost time.",
"[MASK] and peace.",
"crime and [MASK].",
"the sun also [MASK].",
"[MASK] bovary.",
"wuthering [MASK].",
"the importance of [MASK] earnest.",
"a [MASK] named desire.",
"the [MASK] menagerie.",
"oedipus [MASK].",
"[MASK] ado about nothing.",
"one flew over the cuckoo's [MASK].",
"the [MASK] of the rings.",
"the [MASK] in the rye.",
"alice's [MASK] in wonderland.",
"charlie and the [MASK] factory.",
"a tale of two [MASK]."]

def model_tests(zero_frac, layers=[], qkv=[]):

    model_zeros = BertForMaskedLM.from_pretrained("b0rt")

    # convert to string for comparison with layer names
    if len(layers)>0:
        layers = ["layer." + str(l) + "." for l in layers]

    with torch.no_grad(): 
        for i, (name, weight) in enumerate(model_zeros.named_parameters()):
            if weight.shape==torch.Size([768,768]) and name!="cls.predictions.transform.dense.weight":
                if len(layers)==0 or any([l in name for l in layers]):
                    weight.data = weight.data * torch.bernoulli(torch.ones(weight.shape),zero_frac)

    model_zeros.save_pretrained("b0rt_zero")

    plz = pipeline('fill-mask', model='b0rt_zero')
    top_results = []

    for phrase in test_masks:
        top_results.append(pd.DataFrame.from_dict(plz(phrase)))

    return pd.concat(top_results).reset_index(drop=True)

# evaluate performance on masked benchmarks?
# try zeroing out just one of qkv?
top_results = model_tests(0.9)
# top_results.to_csv("top_results.csv",sep="\t")

pct_correct = sum([a==q for a, q in zip(test_answers, list(top_results.iloc[::5]['token_str'].values))])/len(test_answers)
print(pct_correct)
