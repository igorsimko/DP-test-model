from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


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

def sentence_similarity(sentence1, sentence2, trace=False):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        arr_simi_score = []
        for syn2 in synsets2:
            simi_score = synset.path_similarity(syn2)
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




# sentences = [
#     "Dogs are awesome.",
#     "Some gorgeous creatures are felines.",
#     "Dolphins are swimming mammals.",
#     "Cats are beautiful animals.",
# ]
#
# focus_sentence = "Cats are beautiful animals."
#
# for sentence in sentences:
#     print("Similarity(\"%s\", \"%s\") = %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence)))
#     print("Similarity(\"%s\", \"%s\") = %s" % (sentence, focus_sentence, sentence_similarity(sentence, focus_sentence)))
