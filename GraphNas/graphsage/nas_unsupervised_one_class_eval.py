from graphsage.nas_unsupervised_one_class_train import *
actions = [
    [1, 'relu', 1, 'linear'],
    [4, 'sigmoid', 1, 'tanh'],


]
def main(argv=None,action=[4, 'sigmoid', 1, 'tanh']):

    loadArgsForCora(gpu=2,max_total_steps=10**10)
    # loadArgsForReddit(gpu=5)
    global train_data
    print("action:", action)
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, load_walks=True)
    print("Done loading training data..")
    train(train_data, action, regress_fun=run_regression)
if __name__ == '__main__':
    tf.app.run()
