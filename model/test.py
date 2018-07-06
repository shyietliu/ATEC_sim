import tensorflow as tf
import numpy as np
from collections import OrderedDict
from model import Model
import sys
import os
import io
import codecs



reload(sys)
sys.setdefaultencoding('utf8')

# weight = tf.get_variable('w', shape=[8181, 300])
# embed = tf.nn.embedding_lookup(weight, [[1, 2, 3],[1,2,3]])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(embed.shape)
#     print(tf.equal(embed[0], embed[1]).shape)

# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/vocab.txt') as f:
#     full_vocab = f.readlines()
#     vocabulary = list(OrderedDict.fromkeys(full_vocab))
#
# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/full_vocab.txt', 'w') as f:
#     for item in vocabulary:
#         print>> f, item[:-1]
#
# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/test.txt') as f:
#     sub_vocab = f.readlines()
#
# diff = [word for word in full_vocab if word not in sub_vocab]
# pass

# print(-1*np.log(0.5))
# model = Model()
# pred = tf.placeholder(tf.float32, [None, 2])
# y = tf.placeholder(tf.float32, [None, 2])
# f1 = model.compute_f1_score(pred, y)

if os.path.isfile('../dataset/data.txt'):
    with open('../dataset/data.txt', 'r') as f:
        data = f.read().splitlines()

    count_class_0 = 0
    count_class_1 = 0
    to_append_data = []

    for i, data_string in enumerate(data):
        each_data = data_string.split(',')
        label = int(each_data[3])

        if label == 0:
            count_class_0 += 1
        else:
            count_class_1 += 1
            to_append_data.append(data_string)
            to_append_data.append(data_string)
            to_append_data.append(data_string)

        if i%1000 == 0:
            print(i)
        pass

    data = to_append_data
    with codecs.open('../dataset/oversampling_data.txt', 'a', 'utf-8-sig') as f:
    # with open('', 'a') as f:
        for idx, line in enumerate(data):
            f.write((line[0] + ',').decode('utf-8'))
            for item in line[1]:
                f.write(str(item + '/'))
            f.write(',')
            for item in line[2]:
                f.write(str(item + '/'))
            f.write(',')
            f.write(line[3] + '\n')
            if idx % 1000 == 0:
                print('saved {0} data pairs...'.format(idx))
