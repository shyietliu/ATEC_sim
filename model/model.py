import tensorflow as tf


class Model(object):
    def __init__(self):
        """
        Basic Model
        """
        pass

    @staticmethod
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def compute_f1_score(logits, desired):
        """
        Compute f1-score
        :param logits: output logits, shape = [batch_size, output_dim] e.g. [[0.8,  0.2]
                                                                             [0.95, 0.05]]
        :param desired: desired output, shape = [batch_size, output_dim] e.g. [[1, 0]
                                                                               [0, 1]]
        :return: f1-score of the given batch data.
        """
        pred = tf.nn.softmax(logits)
        system_prediction = tf.argmax(pred, 1)
        ground_truth = tf.argmax(desired, 1)

        TP = tf.count_nonzero(system_prediction * ground_truth)
        TN = tf.count_nonzero((system_prediction - 1) * (ground_truth - 1))
        FP = tf.count_nonzero(system_prediction * (ground_truth - 1))
        FN = tf.count_nonzero((system_prediction - 1) * ground_truth)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        # precision = tf.metrics.precision(ground_truth, system_prediction)
        # recall = tf.metrics.recall(desired, pred)
        # TP = tf.reduce_sum(system_prediction*ground_truth)
        # FP = tf.reduce_sum(system_prediction) - TP
        #
        # TN = tf.reduce_sum(tf.cast(tf.logical_not(tf.cast(system_prediction, dtype=tf.bool)), dtype=tf.int64)*
        #                    tf.cast(tf.logical_not(tf.cast(ground_truth, dtype=tf.bool)), dtype=tf.int64))
        # FN = tf.reduce_sum(tf.cast(tf.logical_not(tf.cast(system_prediction, dtype=tf.bool)), dtype=tf.int64)) - TN
        #
        # if (TP+FP) == 0:
        #     print('!!!!!!!')
        #     precision = 0.01
        # else:
        #     precision = TP/(TP+FP)
        # if tf.equal(tf.cast((TP+FN), dtype=tf.int32), tf.constant(0, dtype=tf.int32)):
        #     recall = 0.01
        # else:
        #     print(TP+FN)
        #     recall = TP/(TP+FN)
        # # precision = TP/(TP+FP) if (TP+FP) !=0 else 0.0001
        # # recall = TP/(TP+FN) if (TP+FN) !=0 else 0.0001
        # f1_score = 2 * precision * recall / (precision + recall)
        return f1
        # return [system_prediction, ground_truth], [TP, TN, FP, FN]

    @staticmethod
    def compute_loss(logits, desired):
        desired = tf.cast(desired, dtype=tf.int32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=desired, logits=logits)
        # cross_entropy = -tf.reduce_mean(desired * tf.log(pred))

        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def compute_accuracy(logits, desired):
        pred = tf.nn.softmax(logits)
        print(pred)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(desired, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return accuracy

    @staticmethod
    def attention_layer(x, attn_output_dim=1024):
        """
        :param x: inputs of attention layer, required shape: [batch_size, max_sequence_length, feature_dim
        :param attn_output_dim:
        :return: outputs of attention layer, shape: [batch_size, attn_output_dim]
        """

        align_matrix = tf.matmul(tf.einsum('ijk->ikj', x), x)
        alignment = tf.nn.softmax(align_matrix, 0)
        context_vector = tf.matmul(x, alignment)
        # print(x)
        # print(context_vector)
        # print(tf.concat([tf.reshape(context_vector, [-1, 60 * 128]), x], 1))

        attention_output = tf.layers.dense(tf.concat([tf.reshape(x, [-1, 7680]), tf.reshape(context_vector, [-1, 7680])], 1),  # was T*lstm_hidden_units
                                           attn_output_dim,
                                           activation=tf.nn.tanh)

        return attention_output


if __name__ == '__main__':

    pass

