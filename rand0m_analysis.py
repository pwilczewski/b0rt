
# convert to a jupyter notebook?

import numpy as np
import pandas as pd
import torch
from transformers import pipeline, BertForMaskedLM

# required due to a package conflict in my environment
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# reload the BERT model from package
if False:
    from transformers import BertTokenizer, BertConfig
    model_name = 'bert-base-uncased'
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    model.save_pretrained("b0rt")
    tokenizer.save_pretrained("b0rt")
    config.save_pretrained("b0rt")

mask_answers = ["witch","pride","kill","wrath","young","hundred","monte","search",
                "war","punishment","rises","madame","heights","being","streetcar",
                "glass","rex","much","nest","lord","catcher","adventures",
                "chocolate","cities"]

masked_titles = ["the lion the [MASK] and the wardrobe.",
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

def b0rt_pred(zero_frac=0, layers=[], layer_type='attention_only'):

    model_zeros = BertForMaskedLM.from_pretrained("b0rt")

    # convert to string for comparison with layer names
    if len(layers)>0:
        layers = ["layer." + str(l) + "." for l in layers]
    
    if layer_type=='embedding_only':
        layer_shapes = [torch.Size([30522,768])]
    elif layer_type=='attention_only':
        layer_shapes = [torch.Size([768,768])] 

    with torch.no_grad(): 
        for i, (name, weight) in enumerate(model_zeros.named_parameters()):
            if (weight.shape in layer_shapes) and name!="cls.predictions.transform.dense.weight":
                if len(layers)==0 or any([l in name for l in layers]):
                    weight.data = weight.data * torch.bernoulli(torch.ones(weight.shape),1-zero_frac)

    # save and make predictions for masked titles
    model_zeros.save_pretrained("b0rt_zero")
    plz = pipeline('fill-mask', model='b0rt_zero')
    top_results = []
    for phrase in masked_titles:
        top_results.append(pd.DataFrame.from_dict(plz(phrase)))

    return pd.concat(top_results).reset_index(drop=True)

def calc_accuracy(top_results,n=1):
    return 100*sum([a==q for a, q in zip(mask_answers, list(top_results.iloc[::5]['token_str'].values))])/len(mask_answers)/n

def randomly_zero_params(layer_type='attention_only'):
    results = []
    # calculate accuracy at different degrees of eliminating parameters
    for zero_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pct_correct = []
        for _ in range(10):
            top_preds = b0rt_pred(zero_frac, layer_type=layer_type)
            pct_correct.append(calc_accuracy(top_preds))
        results.append(np.mean(pct_correct))

    results_df = pd.DataFrame(results, index=np.arange(0.05,0.55,step=0.05))
    return results_df

# calculate BERT-base results
baseline_results = b0rt_pred()

# plot average accuracy vs zero fraction
import matplotlib.pyplot as plt

if False:
    results_df = randomly_zero_params(layer_type='attention_only')
    results_df.plot(title="Accuracy vs. fraction of params set to zero",
                    xlabel="Fraction zero", ylabel="Accuracy (%)", legend=False)

def randomly_zero_layers():
    results = []
    for no_layers in range(11):
        pct_correct = []
        for _ in range(10):
            layers_to_zero = np.random.choice(np.arange(12),size=no_layers+1,replace=False).tolist()
            top_preds = b0rt_pred(zero_frac=1, layers=layers_to_zero)
            pct_correct.append(calc_accuracy(top_preds))
        results.append(np.mean(pct_correct))
    
    return pd.DataFrame(results, index=np.arange(1,no_layers+2))

if False:
    results_df = randomly_zero_layers()
    results_df.plot(title="Accuracy vs. number of layers set to zero",
                    xlabel="Number of layers", ylabel="Accuracy (%)", legend=False)

if False:
    results_df = randomly_zero_params(layer_type='embedding_only')
    results_df.plot(title="Accuracy vs. fraction of embedding params set to zero",
                    xlabel="Fraction zero", ylabel="Accuracy (%)", legend=False)

