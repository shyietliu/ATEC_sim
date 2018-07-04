from baseline import MultiLayerPerceptron
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
tf.app.flags.DEFINE_string('data_path', DATA_PATH, "training data dir")
tf.app.flags.DEFINE_integer('epoch', 1000, "Training epochs")
tf.app.flags.DEFINE_float('lr', 1e-3, "Learning rate")
tf.app.flags.DEFINE_boolean('train', True, 'Training status')
tf.app.flags.DEFINE_string('exp_name', 'name', 'experiment name')
tf.app.flags.DEFINE_boolean('save_model', False, 'save model or not')

if __name__ == '__main__':
    model_name = 'MLP_attn_lr_1e-3_epoch_1000'
    model = MultiLayerPerceptron()

    if FLAGS.exp_name == 'default':
        exp_name = model_name
    else:
        exp_name = FLAGS.exp_name

    model.train(FLAGS.epoch,
                exp_name,
                FLAGS.lr,
                save_model=FLAGS.save_model)
    # task = 1,2,3,4,5
    # test = 1, 2, 3, 4
