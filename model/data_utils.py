import numpy as np


class UtilFn(object):

    def __init__(self):
        pass

    @staticmethod
    def one_hot(list_of_index, vocab_size=None, pad=False, max_len=None):
        """
        Making one-hot encoding and padding (optional) for a sequence of indices
        :param list_of_index: a list of word index in vocabulary, e.g. [1, 3, 0]
        :param vocab_size: vocabulary size
        :param pad: padding
        :param max_len: max length of sequence
        :return: a list of one-hot form indices e.g. [[0, 1, 0, 0]  # 1
                                                      [0, 0, 0, 1]  # 3
                                                      [1, 0, 0, 0]  # 0
                                                      [0, 0, 0, 0]  # padding (optional)
                                                      [0, 0, 0, 0]] # padding (optional)
        """
        if vocab_size is None:
            raise Exception('\'vocab_size\' arg is required.')

        if pad:
            if max_len is None:
                raise Exception('If padding, \'max_len\' arg is required.')
            matrix = np.zeros([max_len, vocab_size])

        else:
            matrix = np.zeros([len(list_of_index), vocab_size])
        for i, word_index in enumerate(list_of_index):
            if i > 46:
                print(i)
            matrix[i][word_index] = 1

        return matrix.tolist()