# encoding: utf-8
import csv
import re
import sys
import jieba
import time
from collections import OrderedDict
import os
import numpy as np
import io
import json
from data_utils import UtilFn

reload(sys)
sys.setdefaultencoding('utf8')

# raw train data path
# DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/ATEC/NLP/dataset/atec_nlp_sim_train.csv'
# MODEL_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/ATEC/NLP/model/'
DATA_PATH = '/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train.csv'
# MODEL_PATH = '/Users/shyietliu/python/ATEC/project/NLP/model'
# os.chdir(MODEL_PATH)
np.random.seed(1000)


class DataProvider(object):
    def __init__(self):
        """
        A class for data provider.
        :param path: raw data path, including file name and suffix
        :param batch_size: costumed batch size
        """
        self.path = None
        self.vocabulary = []
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0
        self.train_data_size = None
        self.val_data_size = None
        self.test_data_size = None
        self.batch_size = None
        self.train_cate = 0
        self.val_cate = 0
        self.test_cate = 0
        self.pointer = 0
        self.data = None
        self.data_size = None
        self.ALREADY_LOAD_DATA = 0
        self.ALREADY_LOAD_VOCAB = 0
        self.sequence_max_len = 60
        self.load_vocab()
        self.load_data()

    @property
    def train(self):
        self.data = self.train_data
        self.pointer = self.train_pointer
        self.data_size = self.train_data_size
        self.train_cate = 1
        self.val_cate = 0
        self.test_cate = 0
        return self

    @property
    def val(self):
        self.data = self.val_data
        self.pointer = self.val_pointer
        self.data_size = self.val_data_size
        self.train_cate = 0
        self.val_cate = 1
        self.test_cate = 0
        return self

    @property
    def test(self):
        self.data = self.test_data
        self.pointer = self.test_pointer
        self.data_size = self.test_data_size
        self.train_cate = 0
        self.val_cate = 0
        self.test_cate = 1
        return self

    def pre_process(self, csv_file_path):
        """Clean data and Create a vocabulary, save them to $vocab_path$, and oversampling data"""
        vocab_path = '../dataset/'

        # read data file
        with open(csv_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            data = [row for row in reader]  # data structure: [no., sen1, sen2, label]

        data_size = len(data)
        print(data_size)
        # activate user dict
        jieba.load_userdict('../dataset/user_dict.txt')
        time_start = time.time()
        # sentence cleaning
        for idx, row in enumerate(data):
            # get one sentence pair
            string_1 = row[1]
            string_2 = row[2]

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
            row[1] = jieba.lcut(string_1)
            row[2] = jieba.lcut(string_2)

            # update vocabulary
            self.vocabulary = self.vocabulary + row[1] + row[2]
            time_end = time.time()
            if idx % 1000 == 0:
                print('processed {0} sentence pairs using {1}s'.format(idx, time_end - time_start))


        # delete duplicated
        self.vocabulary = list(OrderedDict.fromkeys(self.vocabulary))

        # v_file_name = 'test_vocab.txt'

        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path)

        # save vocabulary
        # with open(vocab_path+v_file_name, 'w') as f:
        #     for item in self.vocabulary:
        #         print>> f, item

        # save train, val, test data
        with open(vocab_path+'train_data.txt', 'a') as f:
            for i in range(int(data_size*0.8)):
                line = data[i]
                if int(line[3]) == 1:
                    iteration = 3
                else:
                    iteration = 1

                for j in range(iteration):
                    f.write(line[0] + ',')
                    for item in line[1]:
                        f.write(str(item + '/'))
                    f.write(',')
                    for item in line[2]:
                        f.write(str(item + '/'))
                    f.write(',')
                    f.write(line[3] + '\n')
                if i % 1000 == 0:
                    print('saved {0} data pairs...'.format(i))

        with open(vocab_path+'val_data.txt', 'a') as f:
            for i in range(int(data_size*0.8), int(data_size*0.9)):
                line = data[i]
                f.write(line[0] + ',')
                for item in line[1]:
                    f.write(str(item + '/'))
                f.write(',')
                for item in line[2]:
                    f.write(str(item + '/'))
                f.write(',')
                f.write(line[3] + '\n')
                if i % 1000 == 0:
                    print('saved {0} data pairs...'.format(i))

        with open(vocab_path+'test_data.txt', 'a') as f:
            for i in range(int(data_size*0.9), data_size):
                line = data[i]
                f.write(line[0] + ',')
                for item in line[1]:
                    f.write(str(item + '/'))
                f.write(',')
                for item in line[2]:
                    f.write(str(item + '/'))
                f.write(',')
                f.write(line[3] + '\n')
                if i % 1000 == 0:
                    print('saved {0} data pairs...'.format(i))

    def one_hot(self, list_of_index, vocab_size=None, pad=False, max_len=None):
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
        if vocab_size == 2:
            if list_of_index[0] == 0:
                matrix = np.array([1, 0])
            elif list_of_index[1] == 1:
                matrix = np.array([0, 1])

        else:
            padded_seq = np.pad(list_of_index, (0, self.sequence_max_len - len(list_of_index)), 'constant')
            matrix[np.arange(max_len), padded_seq] = 1

        return matrix.tolist()

    def load_vocab(self):
        """Load vocabulary"""
        if not self.ALREADY_LOAD_VOCAB:
            if os.path.isfile('../dataset/vocab.txt'):
                with open('../dataset/vocab.txt', 'r') as f:
                    self.vocabulary = f.read().splitlines()
            else:
                raise Exception('No vocabulary file! Please create vocabulary by running $pre_process$ method first!')
            self.ALREADY_LOAD_VOCAB = 1

    def load_data(self):
        # read data
        if not self.ALREADY_LOAD_DATA:
            if os.path.isfile('../dataset/train_data.txt'):
                print('loading data...')
                with open('../dataset/shuffled_train_data.txt', 'r') as f:
                    self.train_data = f.read().splitlines()
                    self.train_data_size = len(self.train_data)
                with open('../dataset/shuffled_val_data.txt', 'r') as f:
                    self.val_data = f.read().splitlines()
                    self.val_data_size = len(self.val_data)
                with open('../dataset/shuffled_test_data.txt', 'r') as f:
                    self.test_data = f.read().splitlines()
                    self.test_data_size = len(self.test_data)
            else:
                raise Exception('No data file! Please preprocess data by running $pre_process$ method first!')
            self.ALREADY_LOAD_DATA = 1
            # self.split_data()

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
        """
        get batch data
        :returns x1_batch: 2D list of index of the words [batch_size, word_index]
                 x2_batch: 2D list of index of the words [batch_size, word_index]
                 y_batch: label [batch_size] in one-hot form
        """
        # if type(batch_size) is int:
        self.batch_size = batch_size
        # elif batch_size == 'all':
        # #     self.batch_size = len(self.index)
        # total_num_batch = int(self.data_size / self.batch_size)  # the number of batches

        x1_batch = []
        x2_batch = []
        y_batch = []

        for index in range(self.pointer, self.pointer+batch_size):
            each_data = self.data[index].split(',')
            x1 = each_data[1]
            x2 = each_data[2]
            y = [int(each_data[3])]

            x1 = x1.split('/')[:-1]
            x2 = x2.split('/')[:-1]
            x1 = [self.word2idx(x) for x in x1]
            x2 = [self.word2idx(x) for x in x2]
            if data_type == 'one-hot':
                x1_batch.append(self.one_hot(x1, self.vocab_size, pad=True, max_len=self.sequence_max_len))
                x2_batch.append(self.one_hot(x2, self.vocab_size, pad=True, max_len=self.sequence_max_len))
                y_batch.append(self.one_hot(y, 2)[0])
            elif data_type == 'index':
                x1_batch.append(self.padding(x1))
                x2_batch.append(self.padding(x2))
                y_batch.append(y)

        if self.pointer + 2*batch_size > self.data_size:
            print('rest')
            self.pointer = 0  # reset pointer
        else:
            # move pointer to new location
            self.pointer += self.batch_size

        if self.train_cate:
            self.train_pointer = self.pointer
        elif self.val_cate:
            self.val_pointer = self.pointer
        elif self.test_cate:
            self.test_pointer = self.pointer

        return x1_batch, x2_batch, np.array(y_batch).reshape([-1,])

    # def save_json_data(self):
    #     data = {'train': [], 'val': [], 'test': []}
    #
    #     for i, index in enumerate(self.train_index):
    #         time_start = time.time()
    #         data_piece = []  # [sentence1, sentence2, label]
    #         each_data = self.data[index].split(',')
    #         x1 = each_data[1]
    #         x2 = each_data[2]
    #         y = [int(each_data[3])]
    #         print('!!!!!!!!!!!!!!!!', y)
    #
    #         time_mid = time.time()
    #         print(1, time_mid-time_start)
    #         x1 = x1.split('/')[:-1]
    #         x2 = x2.split('/')[:-1]
    #         x1 = [self.word2idx(x) for x in x1]
    #         x2 = [self.word2idx(x) for x in x2]
    #
    #         time_end = time.time()
    #         print(2, time_end - time_mid)
    #
    #         # data_piece.append(self.one_hot(x1, self.vocab_size, pad=True, max_len=60))
    #         # data_piece.append(self.one_hot(x2, self.vocab_size, pad=True, max_len=60))
    #         # data_piece.append(self.one_hot(y, 2, max_len=2)[0])
    #
    #         data_piece.append(x1)
    #         data_piece.append(x2)
    #         data_piece.append(y)
    #
    #         time_xx = time.time()
    #         print(3, time_xx - time_end)
    #
    #         data['train'].append(data_piece)
    #
    #         if i % 10 == 0 and i!=0:
    #             print('processed {0} data pairs'.format(i))
    #
    #
    #
    #     for i, index in enumerate(self.val_index):
    #         data_piece = []  # [sentence1, sentence2, label]
    #         each_data = self.data[index].split(',')
    #         x1 = each_data[1]
    #         x2 = each_data[2]
    #         y = [int(each_data[3])]
    #
    #         x1 = x1.split('/')[:-1]
    #         x2 = x2.split('/')[:-1]
    #         x1 = [self.word2idx(x) for x in x1]
    #         x2 = [self.word2idx(x) for x in x2]
    #
    #         data_piece.append(self.one_hot(x1, self.vocab_size, pad=True, max_len=self.sequence_max_len))
    #         data_piece.append(self.one_hot(x2, self.vocab_size, pad=True, max_len=self.sequence_max_len))
    #         data_piece.append(self.one_hot(y, 2)[0])
    #
    #         data['val'].append(data_piece)
    #
    #         if i % 1000 == 0 and i != 0:
    #             print('processed {0} data pairs'.format(i))
    #
    #             break
    #
    #     for i, index in enumerate(self.test_index):
    #         data_piece = []  # [sentence1, sentence2, label]
    #         each_data = self.data[index].split(',')
    #         x1 = each_data[1]
    #         x2 = each_data[2]
    #         y = [int(each_data[3])]
    #
    #         x1 = x1.split('/')[:-1]
    #         x2 = x2.split('/')[:-1]
    #         x1 = [self.word2idx(x) for x in x1]
    #         x2 = [self.word2idx(x) for x in x2]
    #
    #         data_piece.append(self.one_hot(x1, self.vocab_size, pad=True, max_len=self.sequence_max_len))
    #         data_piece.append(self.one_hot(x2, self.vocab_size, pad=True, max_len=self.sequence_max_len))
    #         data_piece.append(self.one_hot(y, 2)[0])
    #
    #         data['test'].append(data_piece)
    #
    #         if i % 1000 == 0 and i != 0:
    #             print('processed {0} data pairs'.format(i))
    #             break
    #     with io.open('data.json', 'w', encoding='utf-8') as outfile:
    #         json.dump(data, outfile)

    def split_data(self):
        """Randomly split data (7-2-1)"""
        # self.data_size = len(self.data)
        # index = np.random.permutation(self.data_size)
        # self.train_index = index[: int(self.data_size*0.8)]
        # self.val_index = index[int(self.data_size*0.8): int(self.data_size*0.9)]
        # self.test_index = index[int(self.data_size*0.9):]
        pass

    def word2idx(self, word):
        """Convert word to index"""
        return self.vocabulary.index(word)

    def idx2word(self, index):
        """Convert index to word"""
        return self.vocabulary[index]

    @property
    def vocab_size(self):
        """Return the size of vocabulary"""
        return len(self.vocabulary)


if __name__ == '__main__':

    data_provider = DataProvider()

    # data_provider.pre_process('/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train_add.csv')
    x1, x2, y = data_provider.train.next_batch(3)
    v1, v2, vy3 = data_provider.val.next_batch(3)
    xx1, xx2, yy = data_provider.train.next_batch(3)



    # x1, x2, y = data_provider.val.next_batch(10)
    #
    # x1, x2, y = data_provider.test.next_batch(10)
    # data_provider.pre_process('/Users/shyietliu/python/ATEC/project/NLP/dataset/merged.csv')
    # data_provider.pre_process('/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train_add.csv')
    # count_same_meaning = 0


    # count_diff_meaning = 0
    # for i in range(5000):
    #     if i%500 == 0 and i!=0:
    #         print(i)
    #     _, _, y = data_provider.train.next_batch(1)
    #     # print(y)
    #     if y[0][1] == 1:
    #         count_same_meaning += 1
    #     if y[0][0] == 1:
    #         count_diff_meaning += 1
    #
    # print('same meaning number=', count_same_meaning)
    # print('diff meaning number=', count_diff_meaning)
    # train = data_provider.train_index
    # val = data_provider.val_index
    # test = data_provider.test_index
    pass
