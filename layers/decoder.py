import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
from transformer.layers.decoder_layer import DecoderLayer
from transformer.layers.generate_position import generate_positional_encoding


class Decoder(tf.keras.layers.Layer):
	def __init__(self, n, h, vocab_size, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
		# TODO: Update document
		super(Decoder, self).__init__()
		self.n = n
		self.d_model = d_model
		self.decoder_layers = [DecoderLayer(h, d_model, d_ff, activation, dropout_rate, eps) for _ in range(n)]
		self.word_embedding = Embedding(vocab_size, output_dim=d_model)
		self.dropout = Dropout(dropout_rate)

	def call(self, q, encoder_out, is_train, look_ahead_mask=None, padding_mask=None):
		# TODO: Update document
		"""
			Parameters
			----------
			q: tensor
				- target sentence
				- shape: (..., q_length)
			encoder_out: tensor
				- output of encoder
				- shape: (..., k_length, d_model)
			Returns
            ----------
			decoder_out: tensor
				- the new representation of sentence by self attention between target words
				and attention with input words
				- shape: (..., q_length, d_model)
			attentionWeights: dict
				- key: decoder_layer_{$layer_index}_self_attn_weights
					- value: tensor 
						- self_attn_weights
						- shape: (..., q_length, q_length)
				- key: decoder_layer_{$layer_index}_global_attn_weights
					- value: tensor
						- global_attn_weights
						- shape: (..., q_length, k_length)
		"""
		q_length = q.shape[1]
		
		# embedded_q shape: (..., q_length, d_model)
		# TODO: Normalize embedded_q
		embedded_q = self.word_embedding(q)

		embedded_q *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

		# positional_encoding shape: (1, q_length, d_model)
		positional_encoding = generate_positional_encoding(q_length, self.d_model)

		decoder_out = self.dropout(embedded_q + positional_encoding, training=is_train)

		attention_weights = {}

		# Do Attention
		# decoder_out shape: (..., q_length, d_model)
		for i, decoder_layer in enumerate(self.decoder_layers):
			decoder_out, self_attn_weights, global_attn_weights = decoder_layer(decoder_out, encoder_out, is_train, look_ahead_mask, padding_mask)
			attention_weights['decoder_layer_{}_self_attn_weights'.format(i)] = self_attn_weights
			attention_weights['decoder_layer_{}_global_attn_weights'.format(i)] = global_attn_weights
		
		return decoder_out, attention_weights
