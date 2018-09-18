import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class NASSupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos,state_nums, action,concat=True, aggregator_type="mean",
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - action 指导GraphSage生成的参数
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''
        models.GeneralizedModel.__init__(self, **kwargs)
        # 列出聚集器的类型，通过下表进行访问
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
        acts["linear"] = lambda x : x
        acts["softplus"] = tf.nn.softplus
        acts["leaky_relu"] = tf.nn.leaky_relu
        acts["relu6"] = tf.nn.relu6

        # 计算层数
        layers_num = len(action)//state_nums
        self.aggregator_cls = [0]*layers_num
        self.acts = [0]*layers_num
        for i in range(layers_num):
            index = action[i*state_nums]
            self.aggregator_cls[i] = aggregator_types[index] #获得每个聚集器的类别
            self.acts[i] = acts[action[i*state_nums+1]]  # 获取每层的激活函数
        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        # embeds 和 feature都可用作节点的特征；embed随机初始化，在训练过程中被更新
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
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()#建立图模型

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
        # 实例化每层的聚集器
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1 # 如果concat=True，自身向量与邻居向量相连，维度翻倍，之后层的输入维度*2
                # aggregator at current layer
                if layer == len(num_samples) - 1: # 最后一层，激活函数为直接输出，其他层默认为relu
                    aggregator = self.aggregator_cls[layer](dim_mult*dims[layer], dims[layer+1], act=self.acts[layer], # 此处修改，将原来的liner换成激活函数
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls[layer](dim_mult*dims[layer], dims[layer+1],
                            dropout=self.placeholders['dropout'],act=self.acts[layer],#此处修改，修改激活函数
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            # 实施聚集的操作
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims))) # hidden[hop + 1] 是 hidden[hop] 的邻居
                next_hidden.append(h)
            hidden = next_hidden
        #最后一层第一维即为重表示
        return hidden[0], aggregators

    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)#采样，得到样本与支持集大小
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size) #实例化聚集
        dim_mult = 2 if self.concat else 1

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        #全连接层，进行预测
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        # 计算梯度
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        #梯度采集
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        #更新梯度
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        # sigmod 用于多类别， softmax用于单类别
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
