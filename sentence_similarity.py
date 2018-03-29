import operator
import join
import utils
from gensim.models import Word2Vec
from gensim.summarization import keywords
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


def method_most_similiar(category, treshold=3):
    sim_arr = []
    parsed = join.parse_text(category)
    for word in parsed.split(" "):
        synsets = wn.synsets(word)
        # sim_arr.append(word)
        if synsets:
            hypernyms = synsets[0].hypernyms()
            if hypernyms:
                sim_arr.append(hypernyms[0])
    return sim_arr
    sense2freq = {}
    for s in sim_arr:
        freq = 0
        for lemma in s.lemmas():
            freq += lemma.count()
        if freq < treshold:
            sense2freq[s] = freq
    return [x[0].lemmas()[0].name() for x in
            sorted(sorted(sense2freq.items(), key=operator.itemgetter(0)), key=operator.itemgetter(1))]


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
    return [x for x in sorted_x if x[1] > threshold]


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
    """Wind rose as known in ancient Greece, created by the scholar Adamantios Korais around 1796. Without proper rendering support, you may see question marks, boxes, or other symbols. In ancient Greek religion and myth, the Anemoi (Greek: Ἄνεμοι, "Winds")[n 1] were wind gods who were each ascribed a cardinal direction from which their respective winds came (see Classical compass winds), and were each associated with various seasons and weather conditions.""",
    """Promedon (Προμέδων) is a name referring to the following characters in Greek myth or legend:  Promedon, an otherwise unknown figure mentioned in Pausanias' description of Polygnotus' paintings at Lesche in Delphi. Promedon is said to have been depicted leaning against a willow in what seems to be a sacred grove of Persephone, next to such figures as Patroclus and Orpheus. The ancient sources Pausanias claims to have consulted had no uniform opinion concerning Promedon: from some it appeared he was a mere creation of Polygnotus, while others reportedly mentioned him as a music lover who especially favored the singing of Orpheus; thus he could have been believed to be a follower of Orpheus.[1] Promedon of Naxos, a man seduced by his best friend's wife Neaera.[2]""",
    """Sleep temples (also known as dream temples or Egyptian sleep temples) are regarded by some as an early instance of hypnosis over 4000 years ago, under the influence of Imhotep. Imhotep served as Chancellor and as High Priest of the sun god Ra at Heliopolis. He was said to be a son of the ancient Egyptian demiurge Ptah, his mother being a mortal named Khredu-ankh.  Sleep temples were hospitals of sorts, healing a variety of ailments, perhaps many of them psychological in nature. The treatment involved chanting, placing the patient into a trance-like or hypnotic state, and analysing their dreams in order to determine treatment. Meditation, fasting, baths, and sacrifices to the patron deity or other spirits were often involved as well.  Sleep temples also existed in the Middle East and Ancient Greece. In Greece, they were built in honor of Asclepios, the Greek god of medicine and were called Asclepieions. The Greek treatment was referred to as incubation and focused on prayers to Asclepios for healing. A similar Hebrew treatment was referred to as Kavanah and involved focusing on letters of the Hebrew alphabet spelling the name of their god. Mortimer Wheeler unearthed a Roman sleep temple at Lydney Park, Gloucestershire, in 1928, with the assistance of a young J.R.R. Tolkien.[1] """,

]
#
# focus_sentence = "Cats are beautiful animals."
# focus_sentence = "Greek mythology"
# sim_arr = [ get_similarity("Greek mythology", x) for x in sentences]

model = Word2Vec.load('model.bin')

def gramatic_keyword(docs):
    return utils.postprocess_predict(parse_text_keywords(join.parse_text(''.join(docs), join=False),
                                                                      keywords=[x[0] for x in
                                                                                keywords(''.join(docs), scores=True) if
                                                                                x[1] > 0.1]))

def parse_text_keywords(text_phrases, keywords):
    ret_val = []
    for phrase in text_phrases:
        for word in keywords:
            if word in phrase:
                ret_val.append(phrase)
                break

    return ret_val

def get_most_similiar_words_by_category(category_texts, treshold=1):
    # for i in range(len(category_texts)):
    #     category_texts[i] = join.parse_text(category_texts[i])
    flat_arr = []
    arr = []
    for x in category_texts:
        for x1 in category_texts:
            if x != x1:
                xx = get_similarity(x, x1, threshold=treshold)
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

# get_most_similiar_words_by_category(sentences)

# #
# get_most_similiar_words_by_category([keywords((sentences[0])).replace('\n', " "), keywords((sentences[1])).replace('\n', " ")], treshold=0.0)
# n_arr = []
# for sentence in sentences:
#     n_arr.append(' '.join(method_most_similiar(sentence)))
# #     print("Similarity(\"%s\", \"%s\") = %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence)))
# #     print("Similarity(\"%s\", \"%s\") = %s" % (sentence, focus_sentence, sentence_similarity(sentence, focus_sentence)))
#
# print(' '.join(get_most_similiar_words_by_category(n_arr, treshold=0.7)))
