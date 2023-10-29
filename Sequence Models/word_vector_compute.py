# coding: utf-8

import w2v_utils_pytorch

words, word_to_vec_map = w2v_utils_pytorch.read_glove_vecs('data/glove.6B.50d.txt')

# print("cosine_similarity(father, mother) = ", w2v_utils_pytorch.cosine_similarity(father, mother))
# print("cosine_similarity(ball, crocodile) = ",w2v_utils_pytorch.cosine_similarity(ball, crocodile))
# print("cosine_similarity(france - paris, rome - italy) = ",w2v_utils_pytorch.cosine_similarity(france - paris, rome - italy))

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print('{} -> {} <====> {} -> {}'.format(*triad, w2v_utils_pytorch.complete_analogy(*triad, word_to_vec_map)))

# g = word_to_vec_map["woman"] - word_to_vec_map["man"]
# e = "receptionist"
# print("去偏差前{0}与g的余弦相似度为：{1}".format(e, w2v_utils_pytorch.cosine_similarity(word_to_vec_map["receptionist"], g)))
#
# e_debiased = w2v_utils_pytorch.neutralize("receptionist", g, word_to_vec_map)
# print("去偏差后{0}与g的余弦相似度为：{1}".format(e, w2v_utils_pytorch.cosine_similarity(e_debiased, g)))