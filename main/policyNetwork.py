import tensorflow as tf
from tf_agents.networks import network
from tf_agents.distributions import masked
import tensorflow_probability as tfp
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import array_spec
import tensorflow_lattice as tfl
import numpy as np
import pickle
import sys

sys.path.append("../main")
import rlEnvironment

action_id = 0
"""CONQUER policy network"""


def get_curret_id(ids):
    # print(ids)
    global action_id
    action_id = ids


class KGActionDistNet(network.DistributionNetwork):

    def __init__(self,
                 seed_value,
                 input_tensor_spec,
                 output_tensor_spec,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='KGActionDistNet'
                 ):

        self._output_tensor_spec = output_tensor_spec

        super(KGActionDistNet, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=tfp.distributions.Categorical,
            name=name)

        # get same initial values for same seed to make results reproducible
        initializer1 = tf.keras.initializers.GlorotUniform(seed=seed_value)
        initializer2 = tf.keras.initializers.GlorotUniform(seed=(seed_value + 1))
        # define network
        self.dense1 = tf.keras.layers.Dense(768, activation=tf.nn.relu, name="dense1", kernel_initializer=initializer1)
        self.dense2 = tf.keras.layers.Dense(768, name="dense2", kernel_initializer=initializer2)

        self.linear1 = tfl.layers.Linear(num_input_dims=2, use_bias=False)

        self.dist = 0
        self.logits = 0

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec

    def get_distribution(self):
        return self.dist

    def get_logits(self):
        return self.logits

    def load_graph_Embeddings(self, action_id):
        with open("//data1/8steinbi/train_data/embedded_sim_score_vector_model3.pickle", "rb") as emb_file:
            embedded_actions_file = pickle.load(emb_file)
        graph_embedding_list = []

        for startingpoint in action_id:

            length_embedding = len(embedded_actions_file.get(startingpoint))

            embedded_actions = tf.convert_to_tensor(embedded_actions_file.get(startingpoint), dtype=tf.float32)

            if len(embedded_actions) < 1000:
                # zeros = 1000-length_embedding
                zeros = tf.zeros(1000 - length_embedding)
                graph_embedding_list.append(tf.keras.layers.concatenate([embedded_actions, zeros], axis=0))


            else:
                graph_embedding_list.append(embedded_actions)

        if len(action_id) > 1:
            graph_embedding_tensor = tf.stack([point for point in graph_embedding_list])

        else:
            graph_embedding_tensor = tf.convert_to_tensor(graph_embedding_list, dtype=tf.float32)

        return graph_embedding_tensor

    def call(self,
             observations,
             step_type,
             network_state,
             training=False,
             mask=None):

        """get prediction from policy network
        this is called for collecting experience to get the distribution the agent can sample from
        and called once again to get the distribution for a given time step when calculating the loss"""

        is_empty = tf.equal(tf.size(observations), 0)
        if is_empty:
            return 0, network_state

        #action_id = [rlEnvironment.action_id]
        #print('Shape Obsv: ', observations.get_shape().as_list())

        # outer rank will be 0 for one observation, if we have several for calculating the loss it is greater than 1
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        # this is needed because a dense layer expects a batch dimension
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)
        # get individual parts of observation: action mask, optional history embeddings, question and actions
        observations, mask = tf.split(observations, [768, 1], 2)
        #print('Observation first: ', observations)

        if observations.shape[1] == 2001:
            with_history = False
            observations, actions_and_embedding = tf.split(observations, [1, 2000], 1)  # 2x[batchsize,1,768], 1x[batchsize,1000,768]
        elif observations.shape[1] == 2002:
            with_history = True
            history, question, actions_and_embedding = tf.split(observations, [1, 1, 2000],
                                                                1)  # 3x[batchsize,1,768], 1x[batchsize,1000,768]

        if with_history:
            observations = tf.keras.layers.concatenate([history, question], axis=2)  # [batchsize, 1536]


        observations = tf.squeeze(observations, axis=1)
        actions, graph_embeddings = tf.split(actions_and_embedding,[1000,1000 ],1)
        #actions = tf.slice(actions_and_embedding, [0, 0, 0], [1, 1000, 768])
        #graph_embeddings = tf.slice(actions_and_embedding, [0, 1000, 0], [1, 1000, 1])
        #graph_embeddings = tf.squeeze(graph_embeddings, axis=2)


        availableActions = tf.transpose(actions, perm=[0, 2, 1])  # [batchsize,768, 1000]

        mask = tf.squeeze(mask)
        mask_zero = tf.zeros_like(mask)  # (scores>0 and scores<0)
        mask = tf.math.not_equal(mask, mask_zero)
        mask = tf.transpose(mask)
        if with_history:
            mask = mask[:-2]
        else:
            mask = mask[:-1]
        mask = tf.transpose(mask)

        #mask_2 = mask[:1000]

        x = self.dense1(observations)
        out = self.dense2(x)  # [1,768]
        out = tf.expand_dims(out, -1)


        # we multiply actions and output of network and get a matrix where each column is vector for one action, we sum over each column to get score for each action
        scores = tf.multiply(availableActions, out)# [batchsize,1000,1]
        scores = tf.reduce_sum( scores,1)
        '''
        if scores.get_shape().as_list() == [1000, 1000]:
            print('c')
            
            scores_shape = scores.get_shape().as_list()
            mask_2 = tf.slice(mask, begin=[0,0], size=scores_shape)
            transformed_tensor = tf.transpose(scores)  # (1000,1000)
            #transform (1000,1000) row by row by adding with linear layer TransE to Bert Enc.
            scores = tf.map_fn(lambda x : self.transform_tensor(x,graph_embeddings), transformed_tensor)
            scores = tf.squeeze(scores)
            self.logits = scores
            try:
                self.dist = masked.MaskedCategorical(logits=scores, mask=mask_2)
            except:
                transformed_exception_scores= tf.transpose(scores)
                self.dist = masked.MaskedCategorical(logits=transformed_exception_scores, mask=mask_2)
            return self.dist, network_state
        '''
        graph_embeddings = tf.reduce_sum(graph_embeddings, 2)
        x = []
        for l in range(len(graph_embeddings)):
            score = tf.expand_dims(scores[l], 0)
            g = tf.expand_dims(graph_embeddings[l],0)
            scores_2 = tf.concat([score, g], 0)
            scores_2 = tf.transpose(scores_2)
            score_linear= self.linear1(scores_2)
            x.append(score_linear)

        score_linear_layer = tf.convert_to_tensor(x)
        score_linear_layer= tf.reduce_sum(score_linear_layer, 2)
        #tensor_part = tf.map_fn(lambda t: self.transform_tensor(t[0], t[1]),
                                #(scores, graph_embeddings))
        '''
        print(len(scores))
        score_linear_layer=[]
        for startinp in range(len(scores)):
            scores_2 = tf.concat([scores[startinp], graph_embeddings[startinp]], 0)
            scores_2 = tf.transpose(scores_2)
            tensor_part = tf.map_fn(lambda t: self.transform_tensor(t[0], t[1]), (scores[startinp], graph_embeddings[startinp]))
            score_linear_layer.append(tensor_part)
        print(score_linear_layer)
        '''
        #score_linear_layer = self.linear1(scores_2)
        self.logits = score_linear_layer
        # prepare the mask

        #mask_2 = tf.slice(mask, begin=[0], size=[1000]) #the original mask size is larger due to the added 1000 due to the graph embeddings the mask has to be reduced
        if mask.ndim == 1:
            mask= tf.expand_dims(mask, 1)
            mask_2, bin = tf.split(mask, [1000, 1000], 0)
            mask_2 = tf.transpose(mask_2)
        else:
            mask_2, bin = tf.split(mask, [1000, 1000], 1)

        # we convert it to categorical distribution, an action will be sampled from it
        # we use a masking distribution here because we can have less than 1000 valid actions, invalid ones are masked out
        self.dist = masked.MaskedCategorical(logits=score_linear_layer, mask=mask_2)
        return self.dist, network_state

    def transform_tensor2(self, score, transe):
        score_concat = tf.concat([score,transe], 0)
        score_concat = tf.expand_dims(score_concat, 1)
        score_concat = tf.transpose(score_concat)
        l = self.linear1(score_concat)
        return l
    def transform_tensor(self, tensor, transe):

        tensor = tf.expand_dims(tensor, 0)
        if tensor.get_shape().as_list() != [1, 1000]:
            tensor_size = tensor.get_shape().as_list()

            transe_changed = tf.slice(transe, [0,0], [1,tensor_size[-1]])
            concat_tensor = tf.concat([tensor, transe_changed], 0)
            concat_tensor = tf.transpose(concat_tensor)

            return self.linear1(concat_tensor)

        else:
            concat_tensor = tf.concat([tensor,transe], 0)
            concat_tensor = tf.transpose(concat_tensor)

            return self.linear1(concat_tensor)
