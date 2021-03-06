import tensorflow as tf
from model import Model
from tqdm import tqdm
from data_provider import DataProvider
import logger
import time


class LSTM(Model):
    def __init__(self,
                 vocab_size,
                 max_seq_length=60,
                 attn_usage=True
                 ):
        super(LSTM, self).__init__()
        self.hidden_unit_num = 512
        self.output_dim = 2
        self.attn_usage = attn_usage
        self.lstm_hidden_unit_num = 128
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_length

    def embedding_layer(self, x):
        """
        linear embedding
        :param x: input sequence with shape [batch_size, max_sequence_length, vocab_size]
        :return: embedded representation for each word, outputs shape [batch_size, max_sequence_length, embed_dim]
        """
        embed_matrix = tf.Variable(tf.random_normal(shape=[self.vocab_size, 300], mean=0, stddev=1), name='embed_mat')
        embed = tf.einsum('abc,cd->abd', x, embed_matrix)
        return embed

    def representation_extractor(self, x1, x2, keep_probability, norm_gain):

        embedded_x1 = self.embedding_layer(x1)
        embedded_x2 = self.embedding_layer(x2)

        with tf.variable_scope('lstm1'):
            lstm_cell_1 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_unit_num,
                                                                forget_bias=1,
                                                                activation=tf.nn.relu,
                                                                dropout_keep_prob=keep_probability,
                                                                norm_gain=norm_gain)
            #
            # lstm_cell_1_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, input_keep_prob=keep_probability)
            lstm_outputs_1, last_state_1 = tf.nn.dynamic_rnn(lstm_cell_1,
                                                             embedded_x1,
                                                             dtype="float32",
                                                             sequence_length=self.length(embedded_x1))
        with tf.variable_scope('lstm2'):
            lstm_cell_2 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.lstm_hidden_unit_num,
                                                                forget_bias=1,
                                                                activation=tf.nn.relu,
                                                                dropout_keep_prob=keep_probability,
                                                                norm_gain=norm_gain
                                                                )
            # lstm_cell_2_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell_2, input_keep_prob=keep_probability)
            lstm_outputs_2, last_state_2 = tf.nn.dynamic_rnn(lstm_cell_2,
                                                             embedded_x2,
                                                             dtype="float32",
                                                             sequence_length=self.length(embedded_x1))

        if self.attn_usage:
            output_1 = self.attention_layer(lstm_outputs_1, attn_output_dim=512)
            output_2 = self.attention_layer(lstm_outputs_2, attn_output_dim=512)

        hidden = tf.layers.dense(tf.concat([output_1, output_2], 1), 512, tf.nn.relu)
        hidden = tf.nn.dropout(hidden, keep_probability)
        logits = tf.layers.dense(hidden, self.output_dim)

        return logits

    def train(self, epochs, exp_name, lr=1e-4, keep_prob=0.5, normal_gain=0.8, save_model=False, batch_size=10, save_epoch=[]):

        # inputs & outputs format
        # -------------------------------------------------------------------- #
        # x1 = tf.placeholder(tf.float32, [None, self.max_seq_len, self.vocab_size], name='x1')
        # x2 = tf.placeholder(tf.float32, [None, self.max_seq_len, self.vocab_size], name='x2')
        # y = tf.placeholder('float', [None, self.output_dim], name='y')
        # -------------------------------------------------------------------- #

        x1 = tf.placeholder(tf.int32, [None, self.max_seq_len], name='x1')
        x1_one_hot = tf.one_hot(x1, self.vocab_size)
        x2 = tf.placeholder(tf.int32, [None, self.max_seq_len], name='x2')
        x2_one_hot = tf.one_hot(x2, self.vocab_size)
        y = tf.placeholder(tf.int32, [None], name='y')
        y_one_hot = tf.one_hot(y, 2)

        prob = tf.placeholder('float', name='keep_prob')

        norm_gain = tf.placeholder('float', name='norm_gain')

        # construct computation graph
        logits = self.representation_extractor(x1_one_hot, x2_one_hot, prob, normal_gain)
        loss = self.compute_loss(logits, y_one_hot)

        pred = tf.argmax(tf.nn.softmax(logits), 1, name='prediction')
        label = tf.argmax(y_one_hot, 1)

        accuracy = self.compute_accuracy(logits, y_one_hot)

        TP, TN, FP, FN = self.compute_f1_score(logits, y_one_hot)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, name='train_op')

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            data_provider = DataProvider()
            log_saver = logger.LogSaver(exp_name)

            # train
            for epoch in range(epochs):

                # for iteration in range(1):
                for iteration in range(int(112701/batch_size)):

                    # time_start = time.time()
                    batch_x_1, batch_x_2, batch_y = data_provider.train.next_batch(batch_size)
                    sess.run(train_op, feed_dict={x1: batch_x_1,
                                                  x2: batch_x_2,
                                                  y: batch_y,
                                                  prob: keep_prob,
                                                  norm_gain: normal_gain})

                    # validating
                    if iteration % 500 == 0 and iteration != 0:
                        print('running validation ...')
                        train_loss = loss.eval(feed_dict={x1: batch_x_1,
                                                          x2: batch_x_2,
                                                          y: batch_y,
                                                          prob: keep_prob,
                                                          norm_gain: normal_gain})

                        #val_x_1, val_x_2, val_y = data_provider.val.next_batch(1000)
                        #val_acc = accuracy.eval(feed_dict={
                        #                 x1: val_x_1,
                        #                 x2: val_x_2,
                        #                 y: val_y})

                        # Incremental Validation
                        mean_val_acc = 0
                        for i in tqdm(range(100)):
                            # time_start = time.time()
                            val_x_1, val_x_2, val_y = data_provider.val.next_batch(100)
                            val_acc = accuracy.eval(feed_dict={
                                           x1: val_x_1,
                                           x2: val_x_2,
                                           y: val_y,
                                           prob: 1.0,
                                           norm_gain: 1.0}
                                           )

                            mean_val_acc = mean_val_acc + (val_acc - mean_val_acc)/(i+1)

                            # time_end = time.time()
                            # print('evaluate model with batch size=100 takes {0}s'.format(time_end - time_start))
                            # TP, TN, FP, FN

                            TP_score = TP.eval(feed_dict={
                                           x1: val_x_1,
                                           x2: val_x_2,
                                           y: val_y,
                                           prob: 1.0,
                                           norm_gain: 1.0})

                            TN_score = TN.eval(feed_dict={
                                x1: val_x_1,
                                x2: val_x_2,
                                y: val_y,
                                prob: 1.0,
                                norm_gain: 1.0})

                            FP_score = FP.eval(feed_dict={
                                x1: val_x_1,
                                x2: val_x_2,
                                y: val_y,
                                prob: 1.0,
                                norm_gain: 1.0})

                            FN_score = FP.eval(feed_dict={
                                x1: val_x_1,
                                x2: val_x_2,
                                y: val_y,
                                prob: 1.0,
                                norm_gain: 1.0})

                            precision = TP_score/(TP_score+FP_score)
                            recall = TP_score/(TP_score+FN_score)
                            f1 = 2* precision * recall /(precision+recall)
                            print('TP={0}, TN={1}, FP={2}, FN={3}, f1 score = {4}'.format(TP_score,
                                                                                          TN_score,
                                                                                          FP_score,
                                                                                          FN_score,
                                                                                          f1))

                            #prediction = pred.eval(feed_dict={
                            #                x1: val_x_1,
                            #                x2: val_x_2,
                            #                y: val_y,
                            #                prob: 1.0,
                            #                norm_gain: 1.0}
                            #                )
                            # labels = label.eval(feed_dict={
                            #                x1: val_x_1,
                            #                x2: val_x_2,
                            #                y: val_y,
                            #                prob: 1.0,
                            #                norm_gain: 1.0}
                            #                )
                            # print('\nprediction:', prediction, '\nlabel:', labels)
                            # print('f1-score={0}'.format(f1))

                        val_acc = mean_val_acc
                        print('Training {0} epoch, validation accuracy is {1}, training loss is {2}'.format(epoch,
                                                                                                            val_acc,
                                                                                                            train_loss))
                        # save train process
                        log_saver.train_process_saver([epoch, train_loss, val_acc])

                # Model save
                if save_model and epoch+1 in save_epoch:
                    log_saver.model_saver(sess, epoch)

            # evaluate
            mean_test_acc = 0
            for i in tqdm(range(100)):
                test_x_1, test_x_2, test_y = data_provider.test.next_batch(100)
                test_acc = accuracy.eval(feed_dict={
                    x1: test_x_1,
                    x2: test_x_2,
                    y: test_y,
                    prob: 1.0,
                    norm_gain: 1.0}
                )
                mean_test_acc = mean_test_acc + (test_acc - mean_test_acc) / (i + 1)

            test_acc = mean_test_acc

            # test_x_1, test_x_2, test_y = data_provider.test.next_batch(100)
            # test_acc = sess.run(accuracy, feed_dict={x1: test_x_1, x2: test_x_2, y: test_y, prob: 1.0})
            print('test accuracy is {0}'.format(test_acc))
            # save training log
            log_saver.test_result_saver([test_acc])

            


if __name__ == '__main__':
    pass
    #
    # model = LSTM(13250)
    # model.train(epochs=10,
    #             exp_name='test',
    #             lr=1e-4,
    #             keep_prob=0.5,
    #             normal_gain=0.8,
    #             save_model=False,
    #             batch_size=100)
