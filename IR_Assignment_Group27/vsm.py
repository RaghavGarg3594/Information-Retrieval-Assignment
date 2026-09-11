# vsm.py - lnc.ltc vector space model + BM25 ranking

import math
from collections import Counter
from preprocessing import preprocess_text, get_stopwords


def search_vsm(query, index, top_k=10):
    """Rank docs using lnc.ltc weighting. Returns list of (docid, category, title, score)."""
    stop_words = get_stopwords()
    query_terms = preprocess_text(query, stop_words)

    if not query_terms:
        return []

    N = index.N

    # query weights: log-tf * idf (the 'ltc' part)
    query_tf = Counter(query_terms)
    query_weights = {}
    for term, tf in query_tf.items():
        if term in index.inverted_index:
            df = index.inverted_index[term]['df']
            idf = math.log10(N / df)
            w_q = (1 + math.log10(tf)) * idf
            query_weights[term] = w_q

    if not query_weights:
        return []

    # normalize the query vector
    query_norm = math.sqrt(sum(w ** 2 for w in query_weights.values()))
    if query_norm == 0:
        return []
    query_weights = {t: w / query_norm for t, w in query_weights.items()}

    # doc weights: log-tf only, no idf (the 'lnc' part)
    raw_doc_weights = {}
    for term, w_q in query_weights.items():
        postings = index.inverted_index[term]['postings']
        for docid, tf in postings.items():
            w_d = 1 + math.log10(tf)
            if docid not in raw_doc_weights:
                raw_doc_weights[docid] = {}
            raw_doc_weights[docid][term] = w_d

    # need the full doc vector norm (all terms, not just query terms) for cosine
    full_doc_norms = _compute_doc_norms(index)

    # cosine similarity
    scores = {}
    for docid, term_weights in raw_doc_weights.items():
        dot_product = sum(
            term_weights[t] * query_weights[t]
            for t in term_weights
        )
        doc_norm = full_doc_norms.get(docid, 1.0)
        if doc_norm > 0:
            scores[docid] = dot_product / doc_norm

    # sort by score desc, break ties by docid asc
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    results = []
    for docid, score in ranked[:top_k]:
        meta = index.doc_meta[docid]
        results.append((docid, meta['category'], meta['title'], round(score, 6)))
    return results


def _compute_doc_norms(index):
    """L2 norm of each doc's lnc weight vector across ALL terms (not just query terms)."""
    doc_sq_sums = {}
    for term, data in index.inverted_index.items():
        for docid, tf in data['postings'].items():
            w = 1 + math.log10(tf)
            doc_sq_sums[docid] = doc_sq_sums.get(docid, 0) + w ** 2

    return {docid: math.sqrt(sq) for docid, sq in doc_sq_sums.items()}


def search_bm25(query, index, top_k=10, k1=1.2, b=0.75):
    """Okapi BM25 ranking. Same interface as search_vsm."""
    stop_words = get_stopwords()
    query_terms = preprocess_text(query, stop_words)

    if not query_terms:
        return []

    N = index.N
    avgdl = index.avg_doc_length
    query_tf = Counter(query_terms)

    scores = {}
    for term in set(query_terms):
        if term not in index.inverted_index:
            continue
        df = index.inverted_index[term]['df']
        # +1 variant to avoid negative idf for very frequent terms
        idf = math.log10((N - df + 0.5) / (df + 0.5) + 1)

        postings = index.inverted_index[term]['postings']
        for docid, tf in postings.items():
            dl = index.doc_lengths.get(docid, avgdl)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            bm25_term = idf * numerator / denominator
            scores[docid] = scores.get(docid, 0) + bm25_term

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    results = []
    for docid, score in ranked[:top_k]:
        meta = index.doc_meta[docid]
        results.append((docid, meta['category'], meta['title'], round(score, 6)))
    return results
