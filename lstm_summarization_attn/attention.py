from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # features are learned representations of words from encoder
    # hidden are hidden states of the decoder
    def call(self, features, hidden):
        hidden = tf.expand_dims(hidden, axis=-2)
        features = tf.expand_dims(features, axis=-3)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features

        # this is my modification, I think it's better to sum
        # along the second axis, I don't have it confirmed yet
        context_vector = tf.reduce_sum(context_vector, axis=2)
 
        return context_vector, attention_weights
