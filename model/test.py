import tensorflow as tf
import numpy as np
from model import Model
# weight = tf.get_variable('w', shape=[8181, 300])
# embed = tf.nn.embedding_lookup(weight, [[1, 2, 3],[1,2,3]])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(embed.shape)
#     print(tf.equal(embed[0], embed[1]).shape)

# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/vocab.txt') as f:
#     full_vocab = f.readlines()
#
# with open('/Users/shyietliu/python/ATEC/project/NLP/dataset/test.txt') as f:
#     sub_vocab = f.readlines()
#
# diff = [word for word in full_vocab if word not in sub_vocab]
# pass

# print(-1*np.log(0.5))
model = Model()
pred = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 2])
f1 = model.compute_f1_score(pred, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(f1.eval(feed_dict={pred: [[0.8, 0.2],[0.4, 0.6]], y: [[1, 0],[0, 1]]}) )
    # print(recall.eval(feed_dict={pred: [[0.8, 0.2],[0.4, 0.6]], y: [[1, 0],[0, 1]]}))

