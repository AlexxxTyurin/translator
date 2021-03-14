import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization


class RobertaSelfOutput(tf.keras.Model):
    def __init__(self, config):
        super(RobertaSelfOutput, self).__init__()
        self.dense = Dense(config.hidden_size)
        self.LayerNorm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = config.hidden_dropout_prob

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states



