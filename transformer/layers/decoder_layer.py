import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.position_wise_feed_forward_network import ffn

class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, h, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
		# TODO: Update document
		super(DecoderLayer, self).__init__()
		self.masked_mtha = MultiHeadAttention(d_model, h)
		self.mtha = MultiHeadAttention(d_model, h)
		self.feed_forward = ffn(d_ff, d_model, activation)
		self.layernorm1 = LayerNormalization(epsilon=eps)
		self.layernorm2 = LayerNormalization(epsilon=eps)
		self.layernorm3 = LayerNormalization(epsilon=eps)
		self.dropout1 = Dropout(dropout_rate)
		self.dropout2 = Dropout(dropout_rate)
		self.dropout3 = Dropout(dropout_rate)

	def call(self, q, encoder_out, is_train, look_ahead_mask=None, padding_mask=None):
		# TODO: Update document
		"""
			Parameters
			----------
			q: tensor
				query
				shape: (..., q_length, d_model)
				x has d_model to create skip connection with output
			Returns
            ----------
			out: tensor
				the new representation of query by attention between each word
				shape: (..., q_length, d_model)
		"""
		k = v = encoder_out

		# Do self multihead Attention
		# masked_mtha shape: # (..., q_length, d_model)
		masked_mtha_out, self_attn_weights = self.masked_mtha(q, q, q, look_ahead_mask)

		# q shape: # (..., q_length, d_model)
		q = self.layernorm1(q + self.dropout1(masked_mtha_out, training=is_train))

		# Do global multihead Attention
		mtha_out, global_attn_weights = self.mtha(q, k, v, padding_mask)

		# q shape: # (..., q_length, d_model)
		q = self.layernorm2(q + self.dropout2(mtha_out, training=is_train))

		# Do feed forward
		ffn_out = self.feed_forward(q)

		# out shape: # (..., q_length, d_model)
		out = self.layernorm3(q + self.dropout3(ffn_out, training=is_train))

		return out, self_attn_weights, global_attn_weights