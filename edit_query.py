def get_edit_distance(s1, s2, l1, l2):
    if l1 == 0:
        return l2
    if l2 == 0:
        return l1
    if s1[l1 - 1] == s2[l2 - 1]:
        return get_edit_distance(s1, s2, l1 - 1, l2 - 1)

    return 1 + min(get_edit_distance(s1, s2, l1 - 1, l2), get_edit_distance(s1, s2, l1, l2 - 1),
                   get_edit_distance(s1, s2, l1 - 1, l2 - 1))


def get_jaccard_similarity(s1, s2):
    bigrams1 = [s1[i:i + 2] for i in range(len(s1) - 1)]
    bigrams2 = [s2[i:i + 2] for i in range(len(s2) - 1)]
    intersection = list(set(bigrams1) & set(bigrams2))
    union = list(set(bigrams1) | set(bigrams2))

    return len(intersection) / len(union)


def get_closest_words_jaccard(words, s, count):
    jaccard_similarities = [(word, get_jaccard_similarity(s, word)) for word in words]
    jaccard_similarities = sorted(jaccard_similarities, key=lambda x: x[1])
    jaccard_similarities.reverse()
    # print(jaccard_similarities)
    return [jaccard_similarities[i][0] for i in range(count)]


def get_closest_word_edit_distance(words, s):
    distances = [(word, get_edit_distance(word, s, len(word), len(s))) for word in words]
    distances = sorted(distances, key=lambda x: x[1])
    return distances[0][0]

def find_closest_word(words, s, j_count):
    closest_words = get_closest_words_jaccard(words, s, j_count)
    closest_word = get_closest_word_edit_distance(closest_words, s)
    return closest_words, closest_word

def edit_query(dictionary, query):
    edited_query = []
    for qw in query:
        if qw in dictionary:
            edited_query.append(qw)
        else:
            closest_words, closest_word = find_closest_word(dictionary, qw, 30)
            edited_query.append(closest_word)
            print("Closest jaccard words to query word {} : {}".format(qw, closest_words))
            print("Final closest word to query word {} : {}".format(qw, closest_word))

    return edited_query

