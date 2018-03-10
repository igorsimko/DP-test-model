import operator
import numpy as np
import join
from gensim.models import Word2Vec
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def get_similarity(category, text, threshold=0.8):
    sim_arr = {}
    for cat_word in category.lower().split(" "):
        for text_word in text.lower().split(" "):
            if acceptable_word(text_word):
                if cat_word in model and text_word in model:
                    if text_word not in sim_arr:
                        sim_arr.update({text_word: model.wv.similarity(cat_word, text_word)})
                    else:
                        new_sim = model.wv.similarity(cat_word, text_word)
                        if new_sim > sim_arr[text_word]:
                            sim_arr.update({text_word: new_sim})


    sorted_x = sorted(sim_arr.items(), key=operator.itemgetter(1), reverse=True)
    return  [x for x in sorted_x if x[1] > threshold]


def sentence_similarity(sentence1, sentence2, trace=False):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    [x for x in synsets2 if x is not None]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    sim_arr = {}
    category = "Greek mythology".split(" ")

    # For each word in the first sentence
    for synset in synsets1:
        arr_simi_score = []
        for syn2 in synsets2:
            simi_score = synset.wup_similarity(syn2)
            if simi_score is not None:
                arr_simi_score.append(simi_score)

        if (len(arr_simi_score) > 0):
            best = max(arr_simi_score)
            score += best
            count += 1
            # Average the values
        # Get the similarity value of the most similar word in the other sentence
        # best_score = max([synset.path_similarity(ss) for ss in synsets2])
        #
        # # Check that the similarity could have been computed
        # if best_score is not None:
        #     score += best_score
        #     count += 1

    # Average the values
    if count == 0:
        return 0.0
    score /= count
    if trace:
        print("Similarity(\"%s\", \"%s\") = %s" % (sentence1, sentence2, score))

    return score


sentences = [
    """Zeus (/zjuːs/;[3] Greek: Ζεύς Zeús [zdeǔ̯s])[4] is the sky and thunder god in ancient Greek religion, who rules as king of the gods of Mount Olympus. His name is cognate with the first element of his Roman equivalent Jupiter. His mythologies and powers are similar, though not identical, to those of Indo-European deities such as Indra, Jupiter, Perkūnas, Perun, Thor, and Odin.[5][6][7]Zeus is the child of Cronus and Rhea, the youngest of his siblings to be born, though sometimes reckoned the eldest as the others required disgorging from Cronus's stomach. In most traditions, he is married to Hera, by whom he is usually said to have fathered Ares, Hebe, and Hephaestus.[8] At the oracle of Dodona, his consort was said to be Dione, by whom the Iliad states that he fathered Aphrodite.[11] Zeus was also infamous for his erotic escapades. These resulted in many godly and heroic offspring, including Athena, Apollo, Artemis, Hermes, Persephone, Dionysus, Perseus, Heracles, Helen of Troy, Minos, and the Muses.[8]He was respected as an allfather who was chief of the gods[12] and assigned the others to their roles:[13] "Even the gods who are not his natural children address him as Father, and all the gods rise in his presence."[14][15] He was equated with many foreign weather gods, permitting Pausanias to observe "That Zeus is king in heaven is a saying common to all men".[16] Zeus' symbols are the thunderbolt, eagle, bull, and oak. In addition to his Indo-European inheritance, the classical "cloud-gatherer" (Greek: Νεφεληγερέτα, Nephelēgereta)[17] also derives certain iconographic traits from the cultures of the ancient Near East, such as the scepter. Zeus is frequently depicted by Greek artists in one of two poses: standing, striding forward with a thunderbolt leveled in his raised right hand, or seated in majesty.""",
    """The Apollo archetype personifies the aspect of the personality that wants clear definitions, is drawn to master a skill, values order and harmony, and prefers to look at the surface, as opposed to beneath appearances. The Apollo archetype favors thinking over feeling, distance over closeness, objective assessment over subjective intuition.[1][2][3]""",
    """Pandora's box is an artifact in Greek mythology, taken from the myth of Pandora in Hesiod's Works and Days.[1] The container mentioned in the original story is actually a large storage jar but the word was later mistranslated as "box". In modern times an idiom has grown from it meaning “Any source of great and unexpected troubles”,[2] or alternatively “A present which seems valuable but which in reality is a curse”.[3] Later artistic treatments of the fatal container have been very varied, while some literary treatments have focused more on the contents of the idiomatic box than on Pandora herself.""",
    """The Greek Heroic Age, in mythology, is the period between the coming of the Greeks to Thessaly and the Greek return from Troy.[1] It was demarcated as one of the five Ages of Man by Hesiod.[2] The period spans roughly six generations; the heroes denoted by the term are superhuman, though not divine, and are celebrated in the literature of Homer.[1] The Greek heroes can be grouped into an approximate chronology, based on certain events such as of the Argonautic expedition and the Trojan War.""",
    # "Cats are beautiful animals.",
]
#
# focus_sentence = "Cats are beautiful animals."
# focus_sentence = "Greek mythology"
# sim_arr = [ get_similarity("Greek mythology", x) for x in sentences]

model = Word2Vec.load('model.bin')


def get_most_similiar_words_by_category(category_texts, treshold=1):
    for i in range(len(category_texts)):
        category_texts[i] = join.parse_text(category_texts[i])
    flat_arr = []
    arr = []
    for x in category_texts:
        for x1 in category_texts:
            if x != x1:
                xx = get_similarity(x, x1)
                arr.append(xx)
                for i in xx:
                    flat_arr.append(i)


    sorted_x = sorted(flat_arr, key=lambda x: x[1], reverse=True)
    ret = [x[0] for x in sorted_x if x[1] >= treshold]

    ret_val = []
    for x in ret:
        if x not in ret_val:
            ret_val.append(x)
    return ret_val

get_most_similiar_words_by_category(sentences)

#
# for sentence in sentences:
#     print("Similarity(\"%s\", \"%s\") = %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence)))
#     print("Similarity(\"%s\", \"%s\") = %s" % (sentence, focus_sentence, sentence_similarity(sentence, focus_sentence)))
