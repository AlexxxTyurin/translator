import tensorflow as tf


class RobertaLayer(tf.keras.Model):
    def __init__(self, config):
        super(RobertaLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_length_dim = 1
        self.attention = 
        