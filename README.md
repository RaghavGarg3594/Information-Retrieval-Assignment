# Clothing Search Engine — CSD358 Information Retrieval Assignment 1

A search engine built on a corpus of 100 clothing product descriptions. Implements inverted indexing, positional indexing, TF-IDF ranked retrieval (lnc.ltc), phrase search, and proximity search.

## How to Run

### Requirements

- Python 3.8+
- nltk, matplotlib, numpy

Install dependencies:
```
pip install nltk matplotlib numpy
```

### Running the Notebook (main deliverable)

Open and run `main.ipynb` in Jupyter:
```
jupyter notebook main.ipynb
```
Then do **Cell > Run All**.

### Running the CLI

```
python cli.py
```

CLI commands:
```
cotton shirt             → Free-text VSM ranked search
phrase "cotton shirt"    → Exact phrase search
prox stretch denim 3     → Proximity search (within k positions)
bm25 cotton shirt        → BM25 ranked search
quit                     → Exit
```

## Files

| File | Purpose |
|------|---------|
| `main.ipynb` | Main notebook — primary deliverable, run this |
| `preprocessing.py` | Tokenization, stemming (Porter), stopword filtering |
| `indexer.py` | Inverted index and positional index construction |
| `vsm.py` | lnc.ltc VSM ranking + BM25 ranking |
| `positional_search.py` | Phrase search and proximity search using positional index |
| `cli.py` | Command-line interface |
| `corpus_100.txt` | Input corpus (100 clothing documents) |
| `inverted_index.json` | Generated inverted index output |
| `positional_index.json` | Generated positional index output |
