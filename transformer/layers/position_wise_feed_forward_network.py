import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 


def ffn(d_ff=2048, d_model=512, activation='relu'):
    # TODO: Update document
    return Sequential(
        [
            Dense(units=d_ff, activation = activation),
            Dense(d_model)
        ]
    )