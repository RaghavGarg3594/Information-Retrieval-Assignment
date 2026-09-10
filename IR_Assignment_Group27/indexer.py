# indexer.py - builds inverted index + positional index from the corpus

import json
from collections import defaultdict
from preprocessing import parse_corpus, tokenize, preprocess_tokens, get_stopwords, stem


class Index:
    """Parses corpus and builds both index structures in one pass."""

    def __init__(self, corpus_path):
        self.documents = parse_corpus(corpus_path)
        self.N = len(self.documents)
        self.stop_words = get_stopwords()

        # quick lookup for doc metadata by docid
        self.doc_meta = {}
        for doc in self.documents:
            self.doc_meta[doc['docid']] = {
                'category': doc['category'],
                'title': doc['title'],
                'full_text': doc['full_text']
            }

        self.inverted_index = {}
        self.positional_index = {}
        self.doc_lengths = {}       # token count per doc (after preprocessing)
        self.raw_token_counts = {}  # token count before filtering
        self.avg_doc_length = 0.0

        self._build_indices()

    def _build_indices(self):
        inv = defaultdict(lambda: defaultdict(int))      # term -> {docid -> tf}
        pos = defaultdict(lambda: defaultdict(list))     # term -> {docid -> [positions]}

        total_length = 0

        for doc in self.documents:
            docid = doc['docid']
            raw_tokens = tokenize(doc['full_text'])
            self.raw_token_counts[docid] = len(raw_tokens)

            # position counter only increments for tokens that survive filtering
            processed_tokens = []
            position = 0
            for token in raw_tokens:
                if token not in self.stop_words:
                    stemmed = stem(token)
                    processed_tokens.append(stemmed)
                    inv[stemmed][docid] += 1
                    pos[stemmed][docid].append(position)
                    position += 1

            self.doc_lengths[docid] = len(processed_tokens)
            total_length += len(processed_tokens)

        self.avg_doc_length = total_length / self.N if self.N > 0 else 0

        # reshape into the final structure
        for term in inv:
            postings = inv[term]
            self.inverted_index[term] = {
                'df': len(postings),
                'postings': dict(postings)
            }

        for term in pos:
            postings_dict = {}
            for docid, positions in pos[term].items():
                postings_dict[docid] = {
                    'tf': len(positions),
                    'positions': positions
                }
            self.positional_index[term] = {
                'df': len(postings_dict),
                'postings': postings_dict
            }

    def vocabulary(self):
        return sorted(self.inverted_index.keys())

    def vocabulary_size(self):
        return len(self.inverted_index)

    def total_postings(self):
        return sum(entry['df'] for entry in self.inverted_index.values())

    def total_position_entries(self):
        total = 0
        for term_data in self.positional_index.values():
            for doc_data in term_data['postings'].values():
                total += len(doc_data['positions'])
        return total

    def save_inverted_index(self, filepath='inverted_index.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=2, ensure_ascii=False)

    def save_positional_index(self, filepath='positional_index.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.positional_index, f, indent=2, ensure_ascii=False)
