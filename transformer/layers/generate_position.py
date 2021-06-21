import numpy as np
import tensorflow as tf


def createAngleRates(d_model):
    # TODO: Update document
    angles = np.arange(d_model)
    angles[1::2] = angles[0::2]
    angles = 1 / (10000 ** (angles / d_model))
    angles = np.expand_dims(angles, axis=0)
    return angles

def generate_positional_encoding(pos, d_model):
    # TODO: Update document
    angles = createAngleRates(d_model)
    pos = np.expand_dims(np.arange(pos), axis=1)
    pos_angles = pos.dot(angles)
    pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])
    pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])
    pos_angles = np.expand_dims(pos_angles, axis=0)
  
    return tf.cast(pos_angles, dtype=tf.float32)