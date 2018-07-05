import tensorflow as tf
from data_provider import DataProvider

model_path = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/' \
             'exp_log/task1/test_save_model/saved_model/test_save_model.ckpt.meta'
saver = tf.train.import_meta_graph(model_path)
with tf.Session() as sess:

    saver.restore(sess, tf.train.latest_checkpoint('/afs/inf.ed.ac.uk/user/s17/'
                                                   's1700619/E2E_dialog/exp_log/task1/test_save_model/saved_model'))

    print('Model restored!')
    graph = tf.get_default_graph()
    x1 = graph.get_tensor_by_name('x1:0')
    x2 = graph.get_tensor_by_name('x2:0')
    y = graph.get_tensor_by_name('y:0')

    loss = graph.get_tensor_by_name('loss:0')
    acc = graph.get_tensor_by_name('accuracy:0')
    pred = graph.get_tensor_by_name('prediction')

    data_provider = DataProvider('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset')
    batch_x1, batch_x2, batch_y = data_provider.val.next_batch(1000)
    accuracy = sess.run(acc, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})
    print(accuracy)
