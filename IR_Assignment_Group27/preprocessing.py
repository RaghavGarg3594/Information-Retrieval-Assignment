# preprocessing.py - tokenization, stemming, stopword filtering

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)


# words that are normally stopwords but matter for clothing searches
DOMAIN_KEEP_WORDS = {
    'wear',       # "daily wear" vs "festive wear" — changes search meaning
    'fit',        # "slim fit" vs "regular fit"
}

# words that show up in nearly every doc and just add noise
DOMAIN_STOP_WORDS = {
    'designed', 'everyday', 'indian', 'features', 'colour', 'works',
    'well', 'casual', 'office', 'travel', 'styling',
    'depending', 'garment', 'available', 'size', 'standard', 'sizes',
    'suitable', 'comfortable', 'regular', 'use', 'paired', 'common',
    'wardrobe', 'essentials',
}


def get_stopwords():
    """Build final stopword set: NLTK base - domain keepers + domain noise."""
    base = set(stopwords.words('english'))
    base -= DOMAIN_KEEP_WORDS
    base |= DOMAIN_STOP_WORDS
    return base


_stemmer = PorterStemmer()

def stem(token):
    return _stemmer.stem(token)


def parse_corpus(filepath):
    """Parse the XML-style corpus file into a list of doc dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    documents = []
    # each doc is wrapped in <DOC>...</DOC> with DOCID, CATEGORY, TITLE, TEXT
    doc_pattern = re.compile(
        r'<DOC>\s*'
        r'<DOCID>(.*?)</DOCID>\s*'
        r'<CATEGORY>(.*?)</CATEGORY>\s*'
        r'<TITLE>(.*?)</TITLE>\s*'
        r'<TEXT>(.*?)</TEXT>\s*'
        r'</DOC>',
        re.DOTALL
    )

    for match in doc_pattern.finditer(content):
        docid = match.group(1).strip()
        category = match.group(2).strip()
        title = match.group(3).strip()
        text = match.group(4).strip()
        documents.append({
            'docid': docid,
            'category': category,
            'title': title,
            'text': text,
            'full_text': title + '. ' + text  # combine for indexing
        })

    return documents


def tokenize(text):
    """Lowercase + grab alphabetic sequences. Strips punctuation implicitly."""
    return re.findall(r'[a-z]+', text.lower())


def preprocess_tokens(tokens, stop_words=None):
    """Filter stopwords then stem whatever survives."""
    if stop_words is None:
        stop_words = get_stopwords()
    return [stem(t) for t in tokens if t not in stop_words]


def preprocess_text(text, stop_words=None):
    """Full pipeline: tokenize -> filter -> stem."""
    return preprocess_tokens(tokenize(text), stop_words)
