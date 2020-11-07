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


def get_closest_words_jaccard(words, s, j_thresh):
    jaccard_similarities = [(word, get_jaccard_similarity(s, word)) for word in words]
    jaccard_similarities = sorted(jaccard_similarities, key=lambda x: x[1])
    jaccard_similarities.reverse()
    # print(jaccard_similarities)
    close_words = []
    for j_sim in jaccard_similarities:
        if j_sim[1] < j_thresh:
            break
        close_words.append(j_sim[0])
    return close_words


def get_closest_word_edit_distance(words, s):
    distances = [(word, get_edit_distance(word, s, len(word), len(s))) for word in words]
    distances = sorted(distances, key=lambda x: x[1])
    min_distance = distances[0][1]
    close_words = []
    for dist in distances:
        if dist[1] > min_distance:
            break
        close_words.append(dist[0])
    return close_words

def find_closest_word(words, s, j_thresh):
    closest_jaccard_words = get_closest_words_jaccard(words, s, j_thresh)
    closest_distance_words = get_closest_word_edit_distance(closest_jaccard_words, s)
    return closest_jaccard_words, closest_distance_words

def edit_query(dictionary, query):
    edited_query = []
    for qw in query:
        if qw in dictionary:
            edited_query.append(qw)
        else:
            closest_jaccard_words, closest_distance_words = find_closest_word(dictionary, qw, 0.3)
            edited_query.append(closest_distance_words)
            print("Closest jaccard words to query word {} : {}".format(qw, closest_jaccard_words))
            print("Final closest words to query word {} : {}".format(qw, closest_distance_words))

    return edited_query

