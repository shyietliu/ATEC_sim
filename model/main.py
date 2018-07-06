from baseline import MultiLayerPerceptron
from lstm import LSTM
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
tf.app.flags.DEFINE_string('data_path', DATA_PATH, "training data dir")
tf.app.flags.DEFINE_integer('epoch', 1000, "Training epochs")
tf.app.flags.DEFINE_list('save_epoch_list', [1, 3, 5, 7], "after which epoch, save the model")
tf.app.flags.DEFINE_integer('batch_size', 10, "Training epochs")
tf.app.flags.DEFINE_float('lr', 1e-4, "Learning rate")
tf.app.flags.DEFINE_float('keep_prob', 0.5, "Dropout keep probability")
tf.app.flags.DEFINE_float('norm_gain', 0.8, "Gain of layer normalization")
tf.app.flags.DEFINE_boolean('train', True, 'Training status')
tf.app.flags.DEFINE_string('exp_name', 'default', 'experiment name')
tf.app.flags.DEFINE_boolean('save_model', False, 'save model or not')

if __name__ == '__main__':
    model_name = 'LSTM_Attn'
    model = LSTM(13250)

    if FLAGS.exp_name == 'default':
        exp_name = model_name
    else:
        exp_name = FLAGS.exp_name

    model.train(epochs=FLAGS.epoch,
                exp_name=FLAGS.exp_name,
                lr=FLAGS.lr,
                keep_prob=FLAGS.keep_prob,
                normal_gain=FLAGS.norm_gain,
                save_model=FLAGS.save_model,
                batch_size=FLAGS.batch_size,
                save_epoch=FLAGS.save_epoch_list)



# --epoch 10 --exp_name 'LSTM_Attn_lr_1e-4_dropout_05_norm_08' --lr 1e-4 --keep_prob 0.5 --norm_gain 0.8 --save_model True --batch_size 100 --save_epoch_list 0,2