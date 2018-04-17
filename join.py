import nltk
import pandas as pd
from nltk.corpus import stopwords


nltk.download('averaged_perceptron_tagger', 'stopwords', 'wordnet')

# Used when tokenizing words
sentence_re = r'''(?x)      # set flag to allow verbose regexps
      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*            # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()

# Taken from Su Nam Kim Paper...
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""

# grammar = r"""NP: {<NN.>*<JJ.>*}"""

chunker = nltk.RegexpParser(grammar)
stopwords = stopwords.words('english')


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()


def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    # word = stemmer.stem(word)
    word = lemmatizer.lemmatize(word)
    return word


def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):
    for leaf in leaves(tree):
        term = [normalise(w) for w, t in leaf if acceptable_word(w)]
        yield term



def parse_text(text, join=True):
    if not text:
        return ''
    ret_val = []
    toks = nltk.word_tokenize(text)
    postoks = nltk.tag.pos_tag(toks)

    tree = chunker.parse(postoks)
    terms = get_terms(tree)

    for term in terms:
        for word in term:
            if len(word) > 2:
                ret_val.append(word)

    if join:
        return ' '.join(ret_val)
    else:
        return ret_val

# txt = "His mythologies and powers are similar, though not identical, to those of Indo-European deities such as Indra, Jupiter, Perkūnas, Perun, Thor, and Odin"
# print(parse_text("This is quick brown fox. And this is green and black quick dog and my cat is pretty small."))
# print(parse_text(txt))
#