book_answers = ["witch","pride","kill","wrath","young","hundred","monte","search",
                "war","punishment","rises","madame","heights","being","streetcar",
                "glass","rex","much","nest","lord","catcher","adventures",
                "chocolate","cities"]

book_masks = ["[CLS]the lion the [MASK] and the wardrobe.[SEP]",
"[CLS][MASK] and prejudice.[SEP]",
"[CLS]to [MASK] a mockingbird.[SEP]",
"[CLS]the grapes of [MASK].[SEP]",
"[CLS]a portrait of an artist as a [MASK] man.[SEP]",
"[CLS]one [MASK] years of solitude.[SEP]",
"[CLS]the count of [MASK] cristo.[SEP]",
"[CLS]in [MASK] of lost time.[SEP]",
"[CLS][MASK] and peace.[SEP]",
"[CLS]crime and [MASK].[SEP]",
"[CLS]the sun also [MASK].[SEP]",
"[CLS][MASK] bovary.[SEP]",
"[CLS]wuthering [MASK].[SEP]",
"[CLS]the importance of [MASK] earnest.[SEP]",
"[CLS]a [MASK] named desire.[SEP]",
"[CLS]the [MASK] menagerie.[SEP]",
"[CLS]oedipus [MASK].[SEP]",
"[CLS][MASK] ado about nothing.[SEP]",
"[CLS]one flew over the cuckoo's [MASK].[SEP]",
"[CLS]the [MASK] of the rings.[SEP]",
"[CLS]the [MASK] in the rye.[SEP]",
"[CLS]alice's [MASK] in wonderland.[SEP]",
"[CLS]charlie and the [MASK] factory.[SEP]",
"[CLS]a tale of two [MASK].[SEP]"]

def gen_title(w,n):
    return book_masks[n].replace('[MASK]',w)

import pandas as pd
from transformers import pipeline, BertTokenizer

# fine tune embeddings based on this sampling
b0rt_pipe = pipeline('fill-mask', model='b0rt', top_k=1000)
words, token_ids, probs = [], [], []
for n in range(len(book_masks)):
    top1k_tokens = b0rt_pipe(book_masks[n])
    words.append([item['token_str'] for item in top1k_tokens])
    token_ids.append([item['token'] for item in top1k_tokens])
    probs.append([item['score'] for item in top1k_tokens])

import itertools
import numpy as np
# sample 1000 new book titles, excluding the right one
training_data = [np.vectorize(gen_title)(np.random.choice(words[n][1:], size=1000, 
                                  p=probs[n][1:]/np.sum(probs[n][1:])),n).tolist() 
                                  for n in range(len(book_masks))]

training_data = list(itertools.chain(*training_data))
with open('training_data.txt', 'w') as f:
    f.writelines([str(item) + '\n' for item in training_data])

# for looking up token ids in BERT vocab
# tokenizer = BertTokenizer.from_pretrained('b0rt')
# bert_vocab = {v: k for k, v in tokenizer.vocab.items()}

