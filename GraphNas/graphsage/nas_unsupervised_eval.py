from __future__ import division
from __future__ import print_function


from graphsage.nas_unsupervised_train import *

def main(argv=None,action=[2, 'leaky_relu', 3, 'leaky_relu']):
    global train_data
    print("action:",action)
    # 共享训练数据
    # if train_data == None:
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix,load_walks=True)
    print("Done loading training data..")
    train(train_data,action)

if __name__ == '__main__':
    tf.app.run()
