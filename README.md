# SEC 10-K files parsing, embedding building and evaluation.
**Scope:** 10-K files for S&P 500 Companies from 2010-2019<br>
**Goal:** To calibrate Glove embeddings on a financial corpus and see if we can produce something like Loughran-McDonald’s sentiment wordlists or expand it,
so it could be used in the future for other possible purposes as well.<br>
**Steps:**<br>
1.> To achieve this, we cleaned all these 10-K filings through a dynamic pipeline to ingest various types of files.  <br>
2.> Since a single 10-K document is usually a collection of different types of files like html, exhibits, XBRL-related, etc.,
we also analyzed and visualized from the following perspectives: the component of the 10-K files (and its changes for 10 years),
word count and most frequent/important words. <br>
3.> After that, we trained Continuous Bag of Words and Skip Gram model with Glove embedding
as initial weights. We tried to train the models using Yen (the computing resources we have easy access to) and also GPU. 
There’s a boost in training efficiency if we train on GPU but the efficiency is also decent with yen10.<br>
4.> After we trained the model, we came to the evaluation phase where we came up with three perspectives for effectively diagnosing the quality of embedding, 
as well as the difference between Glove, Skip Gram and CBOW models. We first tried to correlate the linguistic uncertainty with stock volatility 
and then used the expansion of words from Loughran-McDonald’s wordlist to see how our model generated expansions look like the original LM list 
both qualitatively and quantitatively.

**Further works:**
We want to approach the financial corpus from different sides so apart from word2vec models, so we also fine-tuned pre-trained BERT and finBert models
to certain/uncertain sentence classification tasks. We hope it could be helpful in the future for further analysis on financial corpus. 

