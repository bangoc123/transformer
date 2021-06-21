
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.layers.multi_head_attention import MultiHeadAttention


d_model = 512
h = 8
batch_size = 64
q_length = 100
k_length = 200 
v_length = 200
d_k = 64
d_v = 64


multiHeadAttention = MultiHeadAttention(d_model, h)

x = tf.constant([
  [0., 1., 2., 2., 5., 1.],
  [2., 2., 4., 2., 0., 3.],
  [0., 2., 4., 2., 1., 2.],
], dtype=tf.float32) # (3, 6)


class MultiHeadAttentionTest(tf.test.TestCase):
  def test_scaled_dot_product_attention(self):
    out, attention_weights = multiHeadAttention.scaled_dot_product_attention(x, x, x)
    self.assertEqual(
      attention_weights.shape,
      (x.shape[0], x.shape[0])
    )

    self.assertEqual(
      out.shape,
      x.shape
    )

  def test_splitting_head(self):
    qw = tf.ones((batch_size, q_length, d_model))
    xs = multiHeadAttention.splitting_head(qw)
    self.assertEqual(
      xs.shape,
      (batch_size, h, q_length, d_model // h)
    )

  def test_call(self):
    q = tf.ones((batch_size, q_length, d_k))
    k = tf.ones((batch_size, k_length, d_k))
    v = tf.ones((batch_size, v_length, d_v))
    final, attention_weights = multiHeadAttention(q, k, v)

    self.assertEqual(
      final.shape, (batch_size, q_length, d_model)
    )
    hd_v = d_model // h
    self.assertEqual(
      attention_weights.shape, (batch_size, h, q_length, k_length)
    )