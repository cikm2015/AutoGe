from graphsage.nas_unsupervised_train import *


tf.app.flags.DEFINE_string('dataset', "Cora", 'dataset used to train.')
def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_embeds)
    train_embeds = scaler.transform(train_embeds)
    test_embeds = scaler.transform(test_embeds)
    dummy = DummyClassifier()
    dummy.fit(train_embeds, train_labels)
    log = SGDClassifier(loss="log", n_jobs=1)
    log.fit(train_embeds, train_labels)
    print("Test scores")
    f1 = f1_score(test_labels, log.predict(test_embeds), average="micro")
    print(f1)
    print("Train scores")
    print(f1_score(train_labels, log.predict(train_embeds), average="micro"))
    # print("Random baseline")
    # print(f1_score(test_labels, dummy.predict(test_embeds), average="micro"))

def evalReddit(action):
    FLAGS.train_prefix = "../example_data/reddit/reddit"
    # FLAGS.train_prefix = "/home/gaoyang/PycharmProject/ProcessDataset/cora"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
    global train_data
    print("action:", action)
    for epoch in range(1, 2, 1):
        FLAGS.epochs = epoch
        # for i in range(100,700,100):
        FLAGS.max_total_steps = 10 ** 10
        print("Loading training data..")
        train_data = load_data(FLAGS.train_prefix, load_walks=True)
        print("Done loading training data..")
        train(train_data, action, regress_fun=run_regression)
def loadArgsForCora(train_prefix = "/home/gaoyang/PycharmProject/ProcessDataset/cora" ,gpu = 4,batch_size=128,validate_batch_size = 128,max_total_steps= 10000):
    print("experiments for cora")
    FLAGS.train_prefix = train_prefix
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    FLAGS.batch_size = batch_size
    FLAGS.validate_batch_size = validate_batch_size
    FLAGS.max_total_steps = max_total_steps
def loadArgsForReddit(train_prefix = "../example_data/reddit/reddit",gpu = 4,max_total_steps = 10**10):
    print("experiments for reddit")
    FLAGS.train_prefix = train_prefix
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    FLAGS.max_total_steps = max_total_steps

# def main(argv=None,action=[1, 'relu6', 0, 'leaky_relu']):
def loadArgsForCiteseer():
    FLAGS.train_prefix = "/home/gaoyang/PycharmProject/ProcessDataset/citeseer"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(4)
    FLAGS.batch_size = 128
    FLAGS.validate_batch_size = 128
    FLAGS.max_total_steps = 7000

def main(argv=None,action=[1, 'relu6', 0, 'leaky_relu'],dataset = None):
    if dataset:
        FLAGS.dataset = dataset
    print(FLAGS.dataset)
    if FLAGS.dataset == "Cora":
        loadArgsForCora()
    elif FLAGS.dataset == "Citeseer":
        loadArgsForCiteseer()
    elif FLAGS.dataset == "Reddit":
        loadArgsForReddit()

    global train_data
    print(action)
    # 共享训练数据
    # if train_data == None:
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix,load_walks=True)
    print("Done loading training data..")
    return train(train_data,action)

if __name__ == '__main__':
    tf.app.run()
