# coding: utf-8

from abc import ABC
import numpy as np
import torch
import w2v_utils_pytorch
import emo_utils
from torch.utils.data import Dataset


class Sentence_Data(Dataset):
    def __init__(self, filename):
        super(Sentence_Data, self).__init__()
        self.max_len = 20
        data, label = emo_utils.read_csv(filename)
        self.label = torch.from_numpy(label)

        self.len = self.label.size()[0]

        words_to_index, index_to_words, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')
        self.embedding = self.pretrained_embedding_layer(word_to_vec_map, words_to_index)
        self.data = self.sentence_to_vec(data, words_to_index=words_to_index)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.len

    def pretrained_embedding_layer(self, word_to_vec_map, word_to_index):
        """
        :param word_to_vec_map:
        :param word_to_index:
        :return:
        """
        vocab_len = len(word_to_index) + 1
        embedding_size = word_to_vec_map["cucumber"].shape[0]

        # initialization
        embedding_matrix = np.zeros((vocab_len, embedding_size))
        for word, index in word_to_index.items():
            embedding_matrix[index, :] = word_to_vec_map[word]

        embedding_matrix = torch.Tensor(embedding_matrix)

        # define embedding layer
        embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix)
        return embedding_layer

    def sentence_to_vec(self, data, words_to_index):
        vec_list = []
        for sentence in data:
            words_index = self.sentences_to_indices(sentence, words_to_index, self.max_len)
            words_index = torch.LongTensor(words_index)
            words_vec = self.embedding(words_index)
            vec_list.append(words_vec)
        return vec_list

    def sentences_to_indices(self, x, words_to_index, max_len):
        """
        :param x
        :param word_to_index
        :param max_len
        :return:
        """
        X_indices = np.zeros(max_len)

        sentences_words = x.lower().split()

        j = 0

        for w in sentences_words:
            X_indices[j] = words_to_index[w]

            j += 1
        return X_indices



def sentences_to_indices(X, word_to_index, max_len):
    """
    :param X
    :param word_to_index
    :param max_len
    :return:
    """
    m = X.shape[0]
    X_indices = np.zeros((max_len,))

    for i in range(m):
        sentences_words = X[i].lower().split()

        j = 0

        for w in sentences_words:
            X_indices[j] = word_to_index[w]

            j += 1
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    :param word_to_vec_map
    :param word_to_index
    :return:
    """
    vocab_len = len(word_to_index) + 1
    embedding_size = word_to_vec_map["cucumber"].shape[0]

    # 初始化嵌入矩阵
    embedding_matrix = np.zeros((vocab_len, embedding_size))
    for word, index in word_to_index.items():
        embedding_matrix[index, :] = word_to_vec_map[word]

    embedding_matrix = torch.Tensor(embedding_matrix)

    # 定义embedding层
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix)
    return embedding_layer


if __name__ == "__main__":
    words_to_index, index_to_words, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')
    embedding = pretrained_embedding_layer(word_to_vec_map, words_to_index)
    sentence = "i love you"
    words = sentence.split()
    words_index = [words_to_index[word] for word in words]
    words_index = torch.LongTensor(words_index)
    words_vec = embedding(words_index)
    words_vec2 = [word_to_vec_map[word] for word in words]
    print(words_vec)
    print(words_vec2)
    # filename = "data/train_emoji.csv"
    # sd = Sentence_Data(filename)
    # print(111)


