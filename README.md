# DSA4213 Assignment 1: Language Models, Embeddings, and Representations on IMDB

Individual assignment for DSA4213 (Natural Language Processing for Data Science), NUS, AY2026/27 Semester 1.
Author: Dziand Dafi Ginandjar (A0354009J).

**Question.** Do the choices that lower a language model's perplexity (architecture, input embeddings) also give better generated text and better features for sentiment classification, and at what cost?

| Part | What varies | What is compared |
|---|---|---|
| I | Architecture | Interpolated trigram, RNN, LSTM, pre-LN Transformer (about 6M parameters each) |
| II | Input embeddings (Transformer) | Random trainable, Skip-gram (own, frozen), GloVe 6B 100d (frozen), random frozen (control) |
| III | Frozen LMs as feature extractors | Mean vs last-token pooling, logistic regression on IMDB sentiment, untrained and TF-IDF controls |

All three parts are in a single notebook, `DSA4213_A1_rev.ipynb`. The report (PDF) is submitted separately.

## Repository contents

```
DSA4213_A1_rev.ipynb   # Parts I-III, run top to bottom, outputs included
README.md
results/               # CSVs written by the notebook (quality, cost, generation metrics, Part II/III runs)
figs/                  # learning_curves.png, part2_curves.png
```

`ckpt/` (cached training runs) is created when the notebook runs and is not tracked.

## Data

- **Source:** IMDB movie reviews, Hugging Face `stanfordnlp/imdb` (Maas et al., 2011). It is downloaded automatically by the `datasets` library; no manual download or credentials are needed.
- **Language-model corpus (Parts I, II):** 30,000 reviews from the `unsupervised` split, shuffled with a fixed seed and split at document level 90/5/5 into 27,000 / 1,500 / 1,500 reviews.
- **Sentiment data (Part III):** labeled `train` and `test` (25k each). Every labeled review whose exact text appears in the LM corpus is removed (223 train, 169 test). The remaining train reviews are split into 19,822 train and 4,955 validation (stratified); the test set has 24,831 reviews and is used once.
- **Pretrained vectors:** GloVe 6B 100d (`glove-wiki-gigaword-100`), downloaded through `gensim.downloader`. It is used only in Part II.

## Preprocessing

- Lower-casing regular-expression tokenizer (words with internal apostrophes kept, punctuation kept, `<br />` removed).
- Vocabulary: the 10,000 most frequent words of the training split plus `<pad>`, `<unk>`, `<bos>`, `<eos>` (|V| = 10,004).
- Sequences truncated to 128 tokens; padding is excluded from the loss and from perplexity.
- The vocabulary, trigram counts, Skip-gram vectors, hyperparameter selection and classifier regularisation use training or validation data only. Each model is evaluated once on its test set.

## Dependencies

Tested on Google Colab with an NVIDIA L4 GPU, Linux, PyTorch 2.11.0 (CUDA 13.0), gensim 4.4.0.

```
pip install datasets gensim scikit-learn scipy
```

PyTorch, NumPy, pandas and matplotlib are pre-installed on Colab. Outside Colab, also install them.

## How to run

1. Open `DSA4213_A1_rev.ipynb` in Google Colab and select a GPU runtime (Runtime > Change runtime type > GPU).
2. In the first code cell, set `QUICK = True` for a tiny smoke test of the whole pipeline, or `QUICK = False` for the real experiment.
3. Run all cells (Runtime > Run all). Data and GloVe are downloaded on the first run.
4. With `RESUME = True`, finished runs are reloaded from `ckpt/`. The cache key is a hash of the configuration, hidden sizes, run name, seed and embedding table, so a changed setting is retrained instead of reused.

Outputs go to `results/` and `figs/`. The last cell checks that evidence exists for each requirement of the assignment brief.

**Runtime** on an L4: about 72 minutes in total (Part I about 33 min, Part II about 38 min), plus about 83 s for Skip-gram and about 4 minutes for the 37 classifier runs.

## Configuration

| | |
|---|---|
| Embedding dimension | 100 |
| RNN / LSTM hidden size | 472 / 414 (1 layer), chosen so total parameters match the Transformer within 0.1% |
| Transformer | 3 layers, width 256, 4 heads, feed-forward 1024, GELU, learned positions, pre-LayerNorm |
| Trigram | Interpolated trigram/bigram/add-one unigram, lambda = (0.2, 0.45, 0.35) |
| Optimiser | AdamW, learning rate 1e-3, weight decay 0.01, 200 warm-up steps then cosine decay |
| Training | Batch 64, 12 epochs, dropout 0.3, gradient clipping 1.0, best-validation checkpoint |
| Skip-gram | gensim, window 5, 5 negative samples, 5 epochs, seed 0, training split only |
| Generation | Greedy and sampled (temperature 0.8, top-k 40), 60 new tokens, 5 prompts |
| Classifier | L2 logistic regression on standardised features; C in {1e-3, ..., 10} chosen on a 20% validation split, then refit on train + validation |

## Random seeds

- Neural language models: seeds 0, 1, 2 (reported as mean and standard deviation).
- Document shuffle and split: `random.Random(0)`.
- Skip-gram: seed 0 (one worker, deterministic).
- Generation: seed 1234, with per-sample offsets.
- Classifier validation split: seed 0.

GPU kernels are not forced to be bit-exact, so a rerun may differ in the last digits.

## Main results (test set, 3 seeds)

| Model | Test perplexity |
|---|---|
| Trigram | 132.6 |
| RNN | 116.0 |
| LSTM | 114.8 |
| Transformer | 105.7 |
| Transformer + frozen Skip-gram (own) | 91.9 |
| Transformer + frozen GloVe | 98.2 |
| Transformer + frozen random (control) | 116.3 |

Sentiment accuracy with frozen mean-pooled features ranges from 70.9% (RNN) to 80.4% (Transformer with Skip-gram embeddings). A TF-IDF logistic-regression baseline reaches 81.9%, higher than every language-model feature. The report has the full tables, controls, error analysis and limitations.

## Responsible use

IMDB and GloVe are used for coursework only and are not redistributed here. No API keys, credentials or private data are included. AI assistance (Claude) was used for coding, debugging and drafting; this is disclosed in the report's Acknowledgements.
