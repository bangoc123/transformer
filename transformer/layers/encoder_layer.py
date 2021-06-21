import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.position_wise_feed_forward_network import ffn

class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, h, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        # TODO: Update document
		super(EncoderLayer, self).__init__()
		self.mtha = MultiHeadAttention(d_model, h)
		self.feed_forward = ffn(d_ff, d_model, activation)
		self.layernorm1 = LayerNormalization(epsilon=eps)
		self.layernorm2 = LayerNormalization(epsilon=eps)
		self.dropout1 = Dropout(dropout_rate)
		self.dropout2 = Dropout(dropout_rate)

	def call(self, x, is_train, mask=None):
		"""
			Parameters
			----------
			x: tensor
				- query
				- shape: (..., q_length, d_model)
				- x has d_model to create skip connection with output
			Returns
            ----------
			out: tensor
				- the new representation of query by attention between each word
				- shape: (..., q_length, d_model)
		"""
		q = x

		# Do self multihead Attention
		# mtha_out shape: # (..., q_length, d_model)
		mtha_out, _ = self.mtha(q, q, q, mask)

		x = self.layernorm1(q + self.dropout1(mtha_out, training=is_train))

		# Do Feed forward

		ffn_out = self.feed_forward(x)


		# (..., q_length, d_model)
		out = self.layernorm2(x + self.dropout2(ffn_out, training=is_train))

		return out