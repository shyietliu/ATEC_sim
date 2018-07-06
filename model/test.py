import tensorflow as tf
import numpy as np
from collections import OrderedDict
from model import Model
import sys
from random import shuffle
import os
import io
import codecs
import csv
import pandas as pd


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
# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train.csv') as csv_file:
#     reader = csv.reader(csv_file, delimiter='\t')
#     data1 = [row for row in reader]

# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/merged.csv') as csv_file:
#     reader = csv.reader(csv_file, delimiter='\t')
#     data2 = [row for row in reader]
#     pass
# data_size = 10
# for i in range(int(data_size*0.8)):
#     print(i)
#
# for i in range(int(data_size*0.8), int(data_size*0.9)):
#     print(i)
#
# for i in range(int(data_size*0.9), data_size):
#     print(i)
# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train.csv') as f:
#     data1 = f.readlines()
#
# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train_add.csv') as f:
#     data2 = f.readlines()
#
# with open('../dataset/merged.csv', 'w') as f:
#     for ele in (data1+data2):
#         f.write(ele)

# a = pd.read_csv("/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train.csv")
# b = pd.read_csv("/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train_add.csv")
# b = b.dropna(axis=1)
# merged = a.merge(b, on='title')
# merged.to_csv("../dataset/merged.csv", index=False)

with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/train_data.txt') as f:
    data = f.readlines()

    # shuffled_data = data

    shuffle(data)

with open('../dataset/shuffled_train_data.txt', 'a') as f:
    for ele in data:
        f.write(ele)

pass



