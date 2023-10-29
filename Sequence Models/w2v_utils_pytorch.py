# coding: utf-8

import numpy as np

def read_glove_vecs(glove_file):
    """
    load glove
    :param glove_file: file path
    :return:
    """
    with open(glove_file, 'r', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return words, word_to_vec_map


def cosine_similarity(u, v):
    """
    :param u:
    :param v:
    :return:
    """
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    distance = np.divide(dot, norm_v * norm_u)

    return distance


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    :param word_a
    :param word_b
    :param word_c
    :param word_to_vec_map
    :return:
    """
    # A -> a
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # word vector
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.keys()

    max_cosine_similarity = -100
    best_word = None

    # search
    for word in words:
        if word in [word_a, word_b, word_c]:
            continue
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))

        if cosine_sim > max_cosine_similarity:
            max_cosine_similarity = cosine_sim
            best_word = word

    return best_word


def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map[word]
    e_biascomponent = np.divide(np.dot(e, g), np.square(np.linalg.norm(g))) * g

    e_debiased = e - e_biascomponent

    return e_debiased
