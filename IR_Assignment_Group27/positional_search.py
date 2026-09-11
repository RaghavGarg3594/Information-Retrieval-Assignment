# positional_search.py - phrase and proximity search using the positional index

from preprocessing import preprocess_text, get_stopwords, stem


def phrase_search(phrase, index):
    """Find docs where the query terms appear consecutively. Returns matches with position evidence."""
    stop_words = get_stopwords()
    terms = preprocess_text(phrase, stop_words)

    if not terms:
        return []

    for t in terms:
        if t not in index.positional_index:
            return []

    # single-word phrase is just a term lookup
    if len(terms) == 1:
        term = terms[0]
        results = []
        for docid, data in sorted(index.positional_index[term]['postings'].items()):
            results.append({
                'docid': docid,
                'category': index.doc_meta[docid]['category'],
                'title': index.doc_meta[docid]['title'],
                'positions': {term: data['positions']}
            })
        return results

    # multi-word: track valid starting positions through each successive term
    first_term = terms[0]
    candidates = {}
    for docid, data in index.positional_index[first_term]['postings'].items():
        candidates[docid] = set(data['positions'])

    for i in range(1, len(terms)):
        term = terms[i]
        if term not in index.positional_index:
            return []
        new_candidates = {}
        for docid, start_positions in candidates.items():
            if docid in index.positional_index[term]['postings']:
                term_positions = set(index.positional_index[term]['postings'][docid]['positions'])
                # keep only starting positions where term_i is at exactly start + i
                valid_starts = {p for p in start_positions if (p + i) in term_positions}
                if valid_starts:
                    new_candidates[docid] = valid_starts
        candidates = new_candidates
        if not candidates:
            return []

    # build results with position evidence for each matched term
    results = []
    for docid in sorted(candidates.keys()):
        start_positions = sorted(candidates[docid])
        pos_evidence = {}
        for i, term in enumerate(terms):
            pos_evidence[term] = sorted({p + i for p in start_positions})
        results.append({
            'docid': docid,
            'category': index.doc_meta[docid]['category'],
            'title': index.doc_meta[docid]['title'],
            'positions': pos_evidence
        })
    return results


def proximity_search(term1, term2, k, index):
    """Find docs where term1 appears before term2 within k positions."""
    s1 = stem(term1.lower())
    s2 = stem(term2.lower())

    if s1 not in index.positional_index or s2 not in index.positional_index:
        return []

    postings1 = index.positional_index[s1]['postings']
    postings2 = index.positional_index[s2]['postings']

    common_docs = set(postings1.keys()) & set(postings2.keys())

    results = []
    for docid in sorted(common_docs):
        positions1 = postings1[docid]['positions']
        positions2 = postings2[docid]['positions']

        matched_p1 = set()
        matched_p2 = set()
        min_dist = float('inf')

        for p1 in positions1:
            for p2 in positions2:
                dist = p2 - p1
                if 0 < dist <= k:
                    matched_p1.add(p1)
                    matched_p2.add(p2)
                    min_dist = min(min_dist, dist)

        if matched_p1 and matched_p2:
            results.append({
                'docid': docid,
                'category': index.doc_meta[docid]['category'],
                'title': index.doc_meta[docid]['title'],
                'distance': min_dist,
                'positions': {
                    s1: sorted(matched_p1),
                    s2: sorted(matched_p2)
                }
            })

    return results


def format_positions(positions_dict):
    """Format position evidence as 'term@[pos1, pos2], ...' for display."""
    parts = []
    for term, positions in positions_dict.items():
        parts.append(f"{term}@{positions}")
    return ', '.join(parts)
