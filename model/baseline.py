import tensorflow as tf
from model import Model
from tqdm import tqdm
from data_provider import DataProvider
import logger


class MultiLayerPerceptron(Model):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden_unit_num = 512
        self.output_dim = 2

    @staticmethod
    def embedding_layer(x):
        """
        linear embedding
        :param x: input sequence with shape [batch_size, max_sequence_length, vocab_size]
        :return: embedded representation for each word, outputs shape [batch_size, max_sequence_length, embed_dim]
        """
        embed_matrix = tf.Variable(tf.random_normal(shape=[8150, 300], mean=0, stddev=0.1), name='embed_mat')
        embed = tf.einsum('abc,cd->abd', x, embed_matrix)
        return embed

    def representation_extractor(self, x1, x2):
        embedded_x1 = self.embedding_layer(x1)
        embedded_x2 = self.embedding_layer(x2)

        hidden_1 = tf.layers.dense(embedded_x1, self.hidden_unit_num, tf.nn.relu)

        hidden_2 = tf.layers.dense(embedded_x2, self.hidden_unit_num, tf.nn.relu)

        # hidden = self.attention_layer(hidden, attn_output_dim=2048)
        logits = tf.layers.dense(tf.reshape(tf.concat([hidden_1, hidden_2], 1), [-1, 45*self.hidden_unit_num*2]),
                                 self.output_dim, name='logits')

        return logits

    def train(self, epochs, exp_name, lr=1e-4, save_model=False):

        # inputs & outputs format
        x1 = tf.placeholder(tf.float32, [None, 45, 8150], name='x1')
        x2 = tf.placeholder(tf.float32, [None, 45, 8150], name='x2')
        y = tf.placeholder('float', [None, self.output_dim], name='y')

        # construct computation graph
        logits = self.representation_extractor(x1, x2)
        loss = self.compute_loss(logits, y)

        accuracy = self.compute_accuracy(logits, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, name='train_op')

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            data_provider = DataProvider('/Users/shyietliu/python/ATEC/project/NLP/dataset/atec_nlp_sim_train.csv')
            log_saver = logger.LogSaver(exp_name)

            # train
            for epoch in range(epochs):
                batch_x_1, batch_x_2, batch_y = data_provider.train.next_batch(10)
                sess.run(train_op, feed_dict={x1: batch_x_1, x2: batch_x_2, y: batch_y})

                # validating
                if epoch % 10 == 0 and epoch != 0:
                    print('running validation ...')
                    train_loss = loss.eval(feed_dict={x1: batch_x_1, x2: batch_x_2, y: batch_y})

                    mean_val_acc = 0
                    for i in tqdm(range(800)):
                        val_x_1, val_x_2, val_y = data_provider.val.next_batch(10)
                        val_acc = accuracy.eval(feed_dict={
                                        x1: val_x_1,
                                        x2: val_x_2,
                                        y: val_y})
                        mean_val_acc = mean_val_acc + (val_acc - mean_val_acc)/(i+1)

                    val_acc = mean_val_acc
                    print('Training {0} epoch, validation accuracy is {1}, training loss is {2}'.format(epoch,
                                                                                                        val_acc,
                                                                                                        train_loss))
                    # save train process
                    log_saver.train_process_saver([epoch, train_loss, val_acc])

            # evaluate
            test_x_1, test_x_2, test_y = data_provider.test.next_batch(10)
            test_acc = sess.run(accuracy, feed_dict={x1: test_x_1, x2: test_x_2, y: test_y})
            print('test accuracy is {0}'.format(test_acc))
            # save training log
            log_saver.test_result_saver([test_acc])

            # Model save
            if save_model:
                log_saver.model_saver(sess)


if __name__ == '__main__':

    DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
    model = MultiLayerPerceptron()
    model.train(20, 'test')
