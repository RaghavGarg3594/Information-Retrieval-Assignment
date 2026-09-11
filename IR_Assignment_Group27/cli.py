# cli.py - interactive search interface for the clothing corpus

import sys
from indexer import Index
from vsm import search_vsm, search_bm25
from positional_search import phrase_search, proximity_search, format_positions


def print_banner():
    print("\n" + "=" * 70)
    print("  Clothing Search Engine — CSD358 Assignment 1")
    print("=" * 70)
    print("Commands:")
    print("  <query>              Free-text ranked search (VSM lnc.ltc)")
    print('  phrase "<words>"     Exact phrase search')
    print("  prox <t1> <t2> <k>  Proximity search (t1 before t2, within k)")
    print("  bm25 <query>        BM25 ranked search")
    print("  quit                 Exit")
    print("-" * 70)


def print_results(results, mode="VSM"):
    if not results:
        print(f"\n  [{mode}] No results found.\n")
        return
    print(f"\n  [{mode}] Top {len(results)} results:\n")
    print(f"  {'Rank':<6}{'DocID':<8}{'Category':<14}{'Title':<45}{'Score'}")
    print(f"  {'-' * 6}{'-' * 8}{'-' * 14}{'-' * 45}{'-' * 10}")
    for i, r in enumerate(results, 1):
        if len(r) == 4:
            docid, category, title, score = r
            print(f"  {i:<6}{docid:<8}{category:<14}{title[:44]:<45}{score:.6f}")
        else:
            print(f"  {i:<6}{r}")


def print_phrase_results(results):
    if not results:
        print("\n  [Phrase] No results found.\n")
        return
    print(f"\n  [Phrase/Proximity] {len(results)} matching document(s):\n")
    for r in results:
        print(f"  {r['docid']} | {r['category']} | {r['title']}")
        pos_str = format_positions(r['positions'])
        print(f"     Positions: {pos_str}")
        if 'distance' in r:
            print(f"     Min distance: {r['distance']}")
        print()


def main():
    print("\n  Loading corpus and building indices...")
    idx = Index('corpus_100.txt')
    print(f"  Indexed {idx.N} documents, {idx.vocabulary_size()} unique terms.\n")

    print_banner()

    while True:
        try:
            user_input = input("\n  Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("  Goodbye!")
            break

        lower = user_input.lower()

        if lower.startswith('phrase '):
            phrase_text = user_input[7:].strip().strip('"').strip("'")
            results = phrase_search(phrase_text, idx)
            print_phrase_results(results)

        elif lower.startswith('prox '):
            parts = user_input.split()
            if len(parts) < 4:
                print("  Usage: prox <term1> <term2> <k>")
                continue
            t1, t2, k_str = parts[1], parts[2], parts[3]
            try:
                k = int(k_str)
            except ValueError:
                print("  Error: k must be an integer.")
                continue
            results = proximity_search(t1, t2, k, idx)
            print_phrase_results(results)

        elif lower.startswith('bm25 '):
            query_text = user_input[5:].strip()
            results = search_bm25(query_text, idx)
            print_results(results, mode="BM25")

        else:
            results = search_vsm(user_input, idx)
            print_results(results, mode="VSM")


if __name__ == '__main__':
    main()
