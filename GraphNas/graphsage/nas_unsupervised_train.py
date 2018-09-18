from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from graphsage.models import  SAGEInfo
from graphsage.nas_unsupervised_models import NASUnsupervisedGraphsage
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data,get_rewards
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '../example_data/ppi', 'name of the object file that stores the training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 50, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 3, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**3, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir(action):
    log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}_{action:s}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            action=str(action),
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr],
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1],
                            feed_dict=feed_dict_val)
        #ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))
    return val_embeddings

def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders
def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import MultiOutputClassifier
    dummy = MultiOutputClassifier(DummyClassifier())
    dummy.fit(train_embeds, train_labels)
    log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=1)
    log.fit(train_embeds, train_labels)

    f1 = 0
    print("val f1_micro:",f1_score(test_labels,log.predict(test_embeds),average='micro'))
    # for i in range(test_labels.shape[1]):
        # print("F1 score", f1_score(test_labels[:,i], log.predict(test_embeds)[:,i], average="micro"))
    # for i in range(test_labels.shape[1]):
    #     print("Random baseline F1 score", f1_score(test_labels[:,i], dummy.predict(test_embeds)[:,i], average="micro"))


def train(train_data,action, test_data=None, regress_fun = run_regression):
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    # Initialize session
    # 定义作用域，不然会与Controller发生冲突
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        with sess.as_default():
            with sess.graph.as_default():
                # Set random seed
                seed = 123
                np.random.seed(seed)
                tf.set_random_seed(seed)

                G = train_data[0]
                features = train_data[1]
                id_map = train_data[2]

                if not features is None:
                    # pad with dummy zero vector
                    features = np.vstack([features, np.zeros((features.shape[1],))])

                context_pairs = train_data[3] if FLAGS.random_context else None
                placeholders = construct_placeholders()
                minibatch = EdgeMinibatchIterator(G,
                          id_map,
                          placeholders, batch_size=FLAGS.batch_size,
                          max_degree=FLAGS.max_degree,
                          num_neg_samples=FLAGS.neg_sample_size,
                          context_pairs=context_pairs)
                adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
                adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

                sampler = UniformNeighborSampler(adj_info)  # 邻居采样，方式为随机重排邻居
                state_nums = 2  # Controller定义的状态数量
                layers_num = len(action) // state_nums  # 计算层数
                layer_infos = []
                # 用于指导最终GNN的生成
                layer_infos.append(SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_1))
                # 用于NAS的无监督GraphSage
                model = NASUnsupervisedGraphsage(placeholders,
                                                 features,
                                                 adj_info,
                                                 minibatch.deg,
                                                 layer_infos=layer_infos,
                                                 action=action,
                                                 state_nums=state_nums,
                                                 model_size=FLAGS.model_size,
                                                 identity_dim = FLAGS.identity_dim,
                                                 logging=True)


                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(log_dir(action), sess.graph)

                # Init variables
                sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

                # Train model

                train_shadow_mrr = None
                shadow_mrr = None

                total_steps = 0
                avg_time = 0.0
                epoch_val_costs = []

                train_adj_info = tf.assign(adj_info, minibatch.adj)
                val_adj_info = tf.assign(adj_info, minibatch.test_adj)
                for epoch in range(FLAGS.epochs):
                    minibatch.shuffle()

                    iter = 0
                    print('Epoch: %04d' % (epoch + 1))
                    epoch_val_costs.append(0)
                    while not minibatch.end():
                        # Construct feed dictionary
                        feed_dict = minibatch.next_minibatch_feed_dict()
                        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                        t = time.time()
                        # Training step
                        outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all,
                                model.mrr, model.outputs1], feed_dict=feed_dict)
                        train_cost = outs[2]
                        train_mrr = outs[5]
                        if train_shadow_mrr is None:
                            train_shadow_mrr = train_mrr#
                        else:
                            train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

                        if iter % FLAGS.validate_iter == 0:
                            # Validation
                            sess.run(val_adj_info.op)
                            val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
                            sess.run(train_adj_info.op)
                            epoch_val_costs[-1] += val_cost
                        if shadow_mrr is None:
                            shadow_mrr = val_mrr
                        else:
                            shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

                        if total_steps % FLAGS.print_every == 0:
                            summary_writer.add_summary(outs[0], total_steps)

                        # Print results
                        avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                        if total_steps % FLAGS.print_every == 0:
                            print("Iter:", '%04d' % iter,
                                  "train_loss=", "{:.5f}".format(train_cost),
                                  "train_mrr=", "{:.5f}".format(train_mrr),
                                  "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
                                  "val_loss=", "{:.5f}".format(val_cost),
                                  "val_mrr=", "{:.5f}".format(val_mrr),
                                  "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                                  "time=", "{:.5f}".format(avg_time))

                        iter += 1
                        total_steps += 1

                        if total_steps > FLAGS.max_total_steps:
                            break

                    if total_steps > FLAGS.max_total_steps:
                            break

                print("Optimization Finished!")
                sess.run(val_adj_info.op)
                # val_losess,val_mrr, duration = incremental_evaluate(sess, model, minibatch,
                #                                                  FLAGS.batch_size)
                # print("Full validation stats:",
                #       "loss=", "{:.5f}".format(val_losess),
                #       "val_mrr=", "{:.5f}".format(val_mrr),
                #       "time=", "{:.5f}".format(duration))

                if FLAGS.save_embeddings:
                    sess.run(val_adj_info.op)

                    embeds = save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(action))

                    if FLAGS.model == "n2v":
                        # stopping the gradient for the already trained nodes
                        train_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                                dtype=tf.int32)
                        test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']],
                                dtype=tf.int32)
                        update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
                        no_update_nodes = tf.nn.embedding_lookup(model.context_embeds,tf.squeeze(train_ids))
                        update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
                        no_update_nodes = tf.stop_gradient(tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
                        model.context_embeds = update_nodes + no_update_nodes
                        sess.run(model.context_embeds)

                        # run random walks
                        from graphsage.utils import run_random_walks
                        nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
                        start_time = time.time()
                        pairs = run_random_walks(G, nodes, num_walks=50)
                        walk_time = time.time() - start_time

                        test_minibatch = EdgeMinibatchIterator(G,
                            id_map,
                            placeholders, batch_size=FLAGS.batch_size,
                            max_degree=FLAGS.max_degree,
                            num_neg_samples=FLAGS.neg_sample_size,
                            context_pairs = pairs,
                            n2v_retrain=True,
                            fixed_n2v=True)

                        start_time = time.time()
                        print("Doing test training for n2v.")
                        test_steps = 0
                        for epoch in range(FLAGS.n2v_test_epochs):
                            test_minibatch.shuffle()
                            while not test_minibatch.end():
                                feed_dict = test_minibatch.next_minibatch_feed_dict()
                                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                                outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all,
                                    model.mrr, model.outputs1], feed_dict=feed_dict)
                                if test_steps % FLAGS.print_every == 0:
                                    print("Iter:", '%04d' % test_steps,
                                          "train_loss=", "{:.5f}".format(outs[1]),
                                          "train_mrr=", "{:.5f}".format(outs[-2]))
                                test_steps += 1
                        train_time = time.time() - start_time
                        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(action), mod="-test")
                        print("Total time: ", train_time+walk_time)
                        print("Walk time: ", walk_time)
                        print("Train time: ", train_time)
    setting = "val"
    tf.reset_default_graph()
    labels = train_data[4]
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]]
    train_labels = np.array([labels[i] for i in train_ids])
    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.array([labels[i] for i in test_ids])
    id_map = {}
    with open(log_dir(action) + "/val.txt") as fp:
        for i, line in enumerate(fp):
            id_map[line.strip()] = i
    train_embeds = embeds[[id_map[str(id)] for id in train_ids]]
    test_embeds = embeds[[id_map[str(id)] for id in test_ids]]

    print("Running regression..")
    regress_fun(train_embeds, train_labels, test_embeds, test_labels)
    #用f1指数替换accuracy，此处未做滑动指数平均
    return get_rewards(val_mrr),val_mrr

moving_acc = 0
train_data = None
structure = [4, 'relu6', 4, 'relu']
structure3 = [4, 'relu', 4, 'tanh']
structure4 = [4, 'relu', 4, 'linear']
structure4 = [4, 'linear', 4, 'tanh']
structure4 = [3, 'tanh', 4, 'linear']


# def main(argv=None,action=[1, 'relu6', 0, 'leaky_relu']):
def main(argv=None,action=[2, 'leaky_relu', 3, 'leaky_relu']):
    global train_data
    print("action:",action)
    # 共享训练数据
    # if train_data == None:
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix,load_walks=True)
    print("Done loading training data..")
    return train(train_data,action)

if __name__ == '__main__':
    tf.app.run()


