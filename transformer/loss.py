import tensorflow as tf
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(loss_object, real, pred):
  mask = tf.math.logical_not(real == 0)
  loss = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss = loss * mask

  return tf.reduce_sum(loss) / tf.reduce_sum(mask)