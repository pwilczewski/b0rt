# Inducing aphasia in large language models

This repo contains my research into removing capabilities from the BERT model. Most ML researchers are focused on adding capabilities to models. My hypothesis is that by trying to remove capabilities we can better understand the structure and behavior of LLMs. Scientists learned a lot about the structure and function of the brain by studying patients with brain injuries - e.g. [Broca's area](https://en.wikipedia.org/wiki/Broca%27s_area). While analogies between artificial neural networks and human brains are imperfect, it is considerably easier to experiment with ANNs.

In my first analysis my goal is to get BERT to forget a selection of famous book titles for masked language modeling. The results of my first pass analysis are available in as a [jupyter notebook](https://github.com/pwilczewski/b0rt/blob/main/ForgettingBookTitles/Evaluate%20Fine%20Tuned%20Models.ipynb).

I am also writing about this research on [substack](https://indiequant.substack.com/p/building-b0rt).
