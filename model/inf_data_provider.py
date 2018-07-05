import jieba
import numpy as np
import csv
from data_utils import UtilFn
import re
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class InferenceDataProvider(UtilFn):
    def __init__(self, inference_data_path):
        super(InferenceDataProvider, self).__init__()
        self.data_path = inference_data_path
        self.data = None
        self.load_data()
        self.data_size = len(self.data)
        self.batch_size = None
        self.index = np.arange(self.data_size)
        self.ALREADY_LOAD_VOCAB = 0
        self.vocabulary = None
        self.sequence_max_len = 60
        self.batch_count = 0
        self.init_jieba()
        self.load_vocab()

    @staticmethod
    def init_jieba():
        jieba.load_userdict('../dataset/user_dict.txt')

    def load_vocab(self):
        """Load vocabulary"""
        if not self.ALREADY_LOAD_VOCAB:
            if os.path.isfile('../dataset/vocab.txt'):
                with open('../dataset/vocab.txt', 'r') as f:
                    self.vocabulary = f.read().splitlines()
                    print ('Loaded vocabulary.')
            else:
                raise Exception('No vocabulary file! Please create vocabulary by running $pre_process$ method first!')
            self.ALREADY_LOAD_VOCAB = 1

    def load_data(self):
        with open(self.data_path) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            self.data = [row for row in reader]  # data structure: [no., sen1, sen2]

    def word2idx(self, word):
        """Convert word to index"""
        if word not in self.vocabulary:
            word = 'UNK'
        return self.vocabulary.index(word)

    def idx2word(self, index):
        """Convert index to word"""
        return self.vocabulary[index]

    def padding(self, seq):
        """
        Padding a sequence of indices
        :param seq: a sequences of indices e.g. [23, 12, 56, 891]
        :return: a padded sequence e.g. [23, 12, 56, 891, 0, 0, 0, 0, 0, 0]

        """
        # padded_seq = np.zeros([len(seq)])
        padded_seq = np.pad(seq, (0, self.sequence_max_len-len(seq)), 'constant')
        # padded_seq.append(row)
        return padded_seq.tolist()

    def next_batch(self, batch_size, data_type='index'):

        if type(batch_size) is int:
            # if self.data_size % batch_size != 0:
            #     raise Exception('batch size must can be divided by {0}'.format(self.data_size))
            self.batch_size = batch_size
        elif batch_size == 'all':
            self.batch_size = self.data_size

        total_num_batch = int(self.data_size / self.batch_size)  # the number of batches

        x1_batch = []
        x2_batch = []

        for index in self.index[self.batch_count:self.batch_count + self.batch_size]:
            # read one specific data pair
            each_data = self.data[index]
            string_1 = each_data[1]  # sentence 1
            string_2 = each_data[2]  # sentence 2

            # split sentence into words by 'jieba'

            # replace '135******09' with 'PHONE_NUM'
            string_1 = re.sub(r'\d{3}\*{6}\d{2}', 'PHONE_NUM', string_1)
            string_2 = re.sub(r'\d{3}\*{6}\d{2}', 'PHONE_NUM', string_2)

            # replace '***' with 'NUM'
            string_1 = re.sub(r'\*{3}', 'NUM', string_1)
            string_2 = re.sub(r'\*{3}', 'NUM', string_2)

            # keep chinese chars only
            string_1 = re.sub(u'[^\u4e00-\u9fa5NUM]+', '', string_1.decode('utf8'))
            string_2 = re.sub(u'[^\u4e00-\u9fa5NUM]*', '', string_2.decode('utf8'))

            # split by JIEBA
            words_in_sent1 = jieba.lcut(string_1)
            words_in_sent2 = jieba.lcut(string_2)

            # convert word into index
            x1 = [self.word2idx(word) for word in words_in_sent1]
            x2 = [self.word2idx(word) for word in words_in_sent2]

            # make one-hot (optional)
            if data_type == 'one-hot':
                x1_batch.append(self.one_hot(x1, self.vocab_size, pad=True, max_len=self.sequence_max_len))
                x2_batch.append(self.one_hot(x2, self.vocab_size, pad=True, max_len=self.sequence_max_len))
            elif data_type == 'index':
                x1_batch.append(self.padding(x1))
                x2_batch.append(self.padding(x2))

        if self.batch_count == total_num_batch - 1:
            self.batch_count = 0  # reset count
        else:
            self.batch_count = self.batch_count + 1

        return x1_batch, x2_batch

    @property
    def vocab_size(self):
        """Return the size of vocabulary"""
        return len(self.vocabulary)


if __name__ == '__main__':
    data_path = '/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train_add.csv'
    data_provider = InferenceDataProvider(data_path)
    x1, x2 = data_provider.next_batch(10, data_type='index')
    pass

