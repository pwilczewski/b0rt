# generates data used for fine tuning
import itertools
import numpy as np
import pandas as pd
from transformers import pipeline

def generate_training_data(path=""):

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

    def gen_title(w,n):
        return masked_titles[n].replace('[MASK]',w)

    # fine tune embeddings based on this sampling
    b0rt_pipe = pipeline('fill-mask', model='bert_orig', top_k=1000)
    words, token_ids, probs = [], [], []
    for n in range(len(masked_titles)):
        top1k_tokens = b0rt_pipe(masked_titles[n])
        words.append([item['token_str'] for item in top1k_tokens])
        token_ids.append([item['token'] for item in top1k_tokens])
        probs.append([item['score'] for item in top1k_tokens])

    # sample 1000 new book titles, excluding the right one
    training_data = [np.vectorize(gen_title)(np.random.choice(words[n][1:], size=1000, 
                                    p=probs[n][1:]/np.sum(probs[n][1:])),n).tolist() 
                                    for n in range(len(masked_titles))]

    # write data to file
    training_data = list(itertools.chain(*training_data))
    with open(path + 'training_data.txt', 'w') as f:
        f.writelines([str(item) + '\n' for item in training_data])

    return

