# imports
from __future__ import print_function
from __future__ import division

import numpy as np
import random
import json
import sys
import os
import time
import tensorflow.compat.v1 as tf
from collections import namedtuple
import math
import time
import pandas as pd
import shutil

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

from graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel 
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
import graphsage.layers as layers
import graphsage.metrics as metrics
from graphsage.input_generator import gen_gsage_inputs

from graphsage.prediction import BipartiteEdgePredLayer
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

############## ESTABLISHING GLOBAL PARAMETERS #################

# Global input parameters that need to remain constant across all embeddings
input_log_device_placement = False # """Whether to log device placement.""" 
input_learning_rate = 0.00001 #, 'initial learning rate.')
input_model_size    = "small" # "Can be big or small; model specific def'ns")

# left to default values in main experiments 
input_epochs = 1 #, 'number of epochs to train.')
input_dropout = 0.0 #, 'dropout rate (1 - keep probability).')
input_weight_decay = 0.0 #, 'weight for l2 loss on embedding matrix.')
input_max_degree = 100 #, 'maximum node degree.')
input_samples_1 = 25 #, 'number of samples in layer 1')
input_samples_2 = 10 #, 'number of users samples in layer 2')
input_random_context = True #, 'Whether to use random context or direct edges')
input_neg_sample_size = 20 #, 'number of negative samples')
input_n2v_test_epochs = 1 #, 'Number of new SGD epochs for n2v.')
input_identity_dim = 0 #, 'Set to positive value to use identity embedding features of that dimension. Default 0.')


#logging, saving, validation settings etc.
input_save_embeddings = True  #, 'whether to save embeddings for all nodes after training')
input_base_log_dir = '.'  #, 'base directory for logging and saving embeddings')
input_validate_iter = 5000  #, "how often to run a validation minibatch.")
input_validate_batch_size = 256  #, "how many nodes per validation sample.")
input_gpu = 1  #, "which gpu to use.")
input_print_every = 50  #, "How often to print training info.")

#random walks variables
WALK_LEN=5
N_WALKS=50


############ MODELS.PY ###############
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name, reuse=True):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    """ A standard multi-layer perceptron """
    def __init__(self, placeholders, dims, categorical=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.categorical = categorical

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']

        #self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=input_learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            #self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            self.loss += input_weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.categorical:
            self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        if self.categorical:
            self.accuracy = metrics.masked_accuracy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                 output_dim=self.dims[1],
                                 act=tf.nn.relu,
                                 dropout=self.placeholders['dropout'],
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(layers.Dense(input_dim=self.dims[1],
                                 output_dim=self.output_dim,
                                 act=lambda x: x,
                                 dropout=self.placeholders['dropout'],
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", identity_dim=10, **kwargs):
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
            - input_learning_rate
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.model_size = model_size
        self.adj_info = adj
        identity_dim = 10
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None:
            identity_dim = 10
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

        #self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=input_learning_rate)

        self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes


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
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
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
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            #num_sampled=FLAGS.neg_sample_size,
            num_sampled=input_neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

           
        # perform "convolution"
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs2, _ = self.aggregate(samples2, [self.features], self.dims, num_samples,
                support_sizes2, aggregators=self.aggregators, concat=self.concat,
                model_size=self.model_size)

        #neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos,
            #FLAGS.neg_sample_size)
        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos,
                                                      input_neg_sample_size)
        
        #self.neg_outputs, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                #neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
                #concat=self.concat, model_size=self.model_size)
        self.neg_outputs, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                neg_support_sizes, batch_size=input_neg_sample_size, aggregators=self.aggregators,
                concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid, 
                bilinear_weights=False,
                name='edge_predict')

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                #self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
                self.loss += input_weight_decay * tf.nn.l2_loss(var)
                

        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs) 
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        #self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, input_neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)


class Node2VecModel(GeneralizedModel):
    def __init__(self, placeholders, dict_size, degrees, name=None,
                 nodevec_dim=50, lr=0.001, **kwargs):
        """ Simple version of Node2Vec/DeepWalk algorithm.

        Args:
            dict_size: the total number of nodes.
            degrees: numpy array of node degrees, ordered as in the data's id_map
            nodevec_dim: dimension of the vector representation of node.
            lr: learning rate of optimizer.
        """

        super(Node2VecModel, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.degrees = degrees
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.batch_size = placeholders['batch_size']
        self.hidden_dim = nodevec_dim

        # following the tensorflow word2vec tutorial
        self.target_embeds = tf.Variable(
                tf.random_uniform([dict_size, nodevec_dim], -1, 1),
                name="target_embeds")
        self.context_embeds = tf.Variable(
                tf.truncated_normal([dict_size, nodevec_dim],
                stddev=1.0 / math.sqrt(nodevec_dim)),
                name="context_embeds")
        self.context_bias = tf.Variable(
                tf.zeros([dict_size]),
                name="context_bias")

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.build()

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            #num_sampled=FLAGS.neg_sample_size,
            num_sampled=input_neg_sample_size,
            unique=True,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        self.outputs1 = tf.nn.embedding_lookup(self.target_embeds, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(self.context_embeds, self.inputs2)
        self.outputs2_bias = tf.nn.embedding_lookup(self.context_bias, self.inputs2)
        self.neg_outputs = tf.nn.embedding_lookup(self.context_embeds, self.neg_samples)
        self.neg_outputs_bias = tf.nn.embedding_lookup(self.context_bias, self.neg_samples)

        self.link_pred_layer = BipartiteEdgePredLayer(self.hidden_dim, self.hidden_dim,
                self.placeholders, bilinear_weights=False)

    def build(self):
        self._build()
        # TF graph management
        self._loss()
        self._minimize()
        self._accuracy()

    def _minimize(self):
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        aff = tf.reduce_sum(tf.multiply(self.outputs1, self.outputs2), 1) + self.outputs2_bias
        neg_aff = tf.matmul(self.outputs1, tf.transpose(self.neg_outputs)) + self.neg_outputs_bias
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        self.loss = loss / tf.cast(self.batch_size, tf.float32)
        tf.summary.scalar('loss', self.loss)
        
    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
       # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        #self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, input_neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)
        
################ UTILS.PY ##################

def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

#random walks variables
WALK_LEN=5
N_WALKS=50

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

################## TRAIN.PY ###################

# log_dir, save_val_embeddings and train
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
tf.disable_eager_execution()
GPU_MEM_FRACTION = 0.8

# Define where to place train output/embeddings
def log_dir():
    log_dir = input_base_log_dir + "/LIVE_2007_2019_date_embeddings"
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

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod, prefix_name, embedding_dim):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    #name = "val"
    name = prefix_name.split("/")[-1]
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
    #np.save(out_dir + '/' + name + '_' + mod + ".npy",  val_embeddings)
    np.save(out_dir + '/' + name + '_' + mod +'_'+str(embedding_dim)+".npy",  val_embeddings)
    
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

def train(train_data, prefix, test_data=None, input_batch_size=50, input_model='graphsage_mean',
         input_dim_1=5, input_dim_2=5, input_max_total_steps=1000):
    input_train_prefix = prefix
    tf.reset_default_graph()
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    embed_dim = 2*input_dim_1

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if input_random_context else None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            placeholders, batch_size=input_batch_size,
            max_degree=input_max_degree, 
            num_neg_samples=input_neg_sample_size,
            context_pairs = context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if input_model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, input_samples_1, input_dim_1),
                            SAGEInfo("node", sampler, input_samples_2, input_dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     model_size=input_model_size,
                                     identity_dim = input_model_size,
                                     logging=True)
    elif input_model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, input_samples_1, 2*input_dim_1),
                            SAGEInfo("node", sampler, input_samples_2, 2*input_dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=input_model_size,
                                     identity_dim = input_identity_dim,
                                     concat=False,
                                     logging=True)

    elif input_model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, input_samples_1, input_dim_1),
                            SAGEInfo("node", sampler, input_samples_2, input_dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     identity_dim = input_identity_dim,
                                     aggregator_type="seq",
                                     model_size=input_model_size,
                                     logging=True)

    elif input_model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, input_samples_1, input_dim_1),
                            SAGEInfo("node", sampler, input_samples_2, input_dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=input_model_size,
                                     identity_dim = input_identity_dim,
                                     logging=True)
    elif input_model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, input_samples_1, input_dim_1),
                            SAGEInfo("node", sampler, input_samples_2, input_dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=input_model_size,
                                     identity_dim = input_identity_dim,
                                     logging=True)

    elif input_model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                                       minibatch.deg,
                                       #2x because graphsage uses concat
                                       nodevec_dim=2*input_dim_1,
                                       lr=input_learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=input_log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
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
    for epoch in range(input_epochs): 
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            #print('flagg')
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: input_dropout})

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

            if iter % input_validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=input_validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

            if total_steps % input_print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % input_print_every == 0:
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

            if total_steps > input_max_total_steps:
                break

        if total_steps > input_max_total_steps:
                break
    print('Iterations used= '+ str(total_steps))
    print("Optimization Finished!")
    if input_save_embeddings:
        sess.run(val_adj_info.op)

        save_val_embeddings(sess, model, minibatch, input_validate_batch_size, log_dir(), mod=input_model, prefix_name=input_train_prefix, embedding_dim=embed_dim)

        if input_model == "n2v":
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
                placeholders, batch_size=input_batch_size,
                max_degree=input_max_degree, 
                num_neg_samples=input_neg_sample_size,
                context_pairs = pairs,
                n2v_retrain=True,
                fixed_n2v=True)
            
            start_time = time.time()
            print("Doing test training for n2v.")
            test_steps = 0
            for epoch in range(input_n2v_test_epochs):
                test_minibatch.shuffle()
                while not test_minibatch.end():
                    feed_dict = test_minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: input_dropout})
                    outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all, 
                        model.mrr, model.outputs1], feed_dict=feed_dict)
                    if test_steps % input_print_every == 0:
                        print("Iter:", '%04d' % test_steps, 
                              "train_loss=", "{:.5f}".format(outs[1]),
                              "train_mrr=", "{:.5f}".format(outs[-2]))
                    test_steps += 1
            train_time = time.time() - start_time
            save_val_embeddings(sess, model, minibatch, input_validate_batch_size, log_dir(), mod="-test", prefix_name=input_train_prefix,  embedding_dim=embed_dim)
            print("Total time: ", train_time+walk_time)
            print("Walk time: ", walk_time)
            print("Train time: ", train_time)








