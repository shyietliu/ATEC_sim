from __future__ import print_function
import os
import tensorflow as tf


class LogSaver(object):
    def __init__(self, exp_name, exp_path=None):
        """
        Auto save log to '../exp_log'
        Make sure your working directory is compatible with current directory.
        :param exp_name:
        :param log_path:
        """
        if exp_path is None:
            # the path of a specific experiment
            self.exp_path = os.path.join('../exp_log', exp_name)
        else:
            self.exp_path = exp_path
        print('Log file will be saved in \'{0}\''.format(self.exp_path))
        self.exp_name = exp_name
        self.log_path = None  # log file path exclude file name
        self.model_path = None  # model file path exclude file name
        self.log_file_path = None  # log file path include file name
        self.model_file_path = None  # model file path include file name
        self._init_log()

    def _init_log(self):
        """
        Setting the path of saved log
        :param task_cat:
        :return:
        """
        # set log file path
        log_file_name = 'Log_' + self.exp_name + '.txt'
        self.log_path = os.path.join(self.exp_path, 'log')
        self.log_file_path = os.path.join(self.log_path, log_file_name)

        # set model file path
        self.model_path = os.path.join(self.exp_path, 'model')
        model_file_name = self.exp_name + '.ckpt'
        self.model_file_path = os.path.join(self.model_path, model_file_name)

    def train_process_saver(self, information):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(self.log_file_path, 'a') as f:
            print('Epoch,{0}, Train loss,{1}, Val_acc,{2}'.format(information[0],
                                                                  information[1],
                                                                  information[2]), file=f)

    def test_result_saver(self, information):
        if not os.path.exists(self.log_file_path):
            os.makedirs(self.log_file_path)
        with open(self.log_file_path, 'a') as f:
            print('Test acc {0}'.format(information[0]), file=f)

    def model_saver(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.model_file_path)

if __name__ == '__main__':
    log_saver = LogSaver('exp')