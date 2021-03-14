import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Softmax
from bert_modelling.utils import swap_axes
import math


class RobertaSelfAttention(tf.keras.Model):
    def __init__(self, config):
        super(RobertaSelfAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(
                f"""The hidden size({config.hidden_size}) is not the multiple of 
                    the number of attention heads({config.num_attention_heads})"""
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Dense(self.all_head_size)
        self.key = Dense(self.all_head_size)
        self.value = Dense(self.all_head_size)

        self.dropout = Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = tf.reshape(x, shape=new_x_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,
             hidden_states,
             attention_mask=None,
             head_mask=None,
             encoder_hidden_states=None,
             encoder_attention_mask=None,
             pask_key_value=None,
             output_attentions=False):

        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and pask_key_value is not None:
            key_layer = pask_key_value[0]
            value_layer = pask_key_value[1]
            attention_mask = encoder_attention_mask

        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask

        elif pask_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.key(hidden_states))
            key_layer = tf.concat([pask_key_value[0], key_layer], dim=2)
            value_layer = tf.concat([pask_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.key(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            pask_key_value = (key_layer, value_layer)

        attention_scores = tf.matmul(query_layer, swap_axes(key_layer, [-1, -2]))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = tf.reshape(tf.range(0, seq_length, dtype=tf.int64), shape=(-1, 1))
            position_ids_r = tf.reshape(tf.range(0, seq_length, dtype=tf.int64), shape=(1, -1))
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = tf.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = tf.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = tf.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores =attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = Softmax()(attention_scores)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = tf.reshape(context_layer, shape=new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (pask_key_value,)

        return outputs





        



