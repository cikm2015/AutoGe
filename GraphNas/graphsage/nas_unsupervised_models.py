import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from graphsage.models import BipartiteEdgePredLayer
flags = tf.app.flags
FLAGS = flags.FLAGS


class NASUnsupervisedGraphsage(models.SampleAndAggregate):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos,action,state_nums, concat=True, aggregator_type="mean",
                 model_size="small", identity_dim=0,
                 **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        models.GeneralizedModel.__init__(self, **kwargs)

        aggregator_types = []
        aggregator_types.append(MeanAggregator)
        aggregator_types.append(SeqAggregator)
        aggregator_types.append(MeanPoolingAggregator)
        aggregator_types.append(MaxPoolingAggregator)
        aggregator_types.append(GCNAggregator)

        acts = {}
        acts["sigmoid"] = tf.nn.sigmoid
        acts["tanh"] = tf.nn.tanh
        acts["relu"] = tf.nn.relu
        acts["linear"] = lambda x: x
        acts["softplus"] = tf.nn.softplus
        acts["leaky_relu"] = tf.nn.leaky_relu
        acts["relu6"] = tf.nn.relu6

        # 计算层数
        layers_num = len(action) // state_nums
        self.aggregator_cls = [0] * layers_num
        self.acts = [0] * layers_num
        for i in range(layers_num):
            index = action[i * state_nums]
            self.aggregator_cls[i] = aggregator_types[index]  # 获得每个聚集器的类别
            self.acts[i] = acts[action[i * state_nums + 1]]  # 获取每层的激活函数

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat

        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
                  aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls[layer](dim_mult * dims[layer], dims[layer + 1], act=self.acts[layer],
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls[layer](dim_mult * dims[layer], dims[layer + 1],act=self.acts[layer],
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              dim_mult * dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators




