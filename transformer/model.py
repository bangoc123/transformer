
from transformer.layers.encoder import Encoder
from transformer.layers.decoder import Decoder
from tensorflow.keras.layers import Dense
from transformer.layers.generate_mask import generate_mask
import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

def cal_acc(real, pred):
	accuracies = tf.equal(real, tf.argmax(pred, axis=2))

	mask = tf.math.logical_not(tf.math.equal(real, 0))
	accuracies = tf.math.logical_and(mask, accuracies)

	accuracies = tf.cast(accuracies, dtype=tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class Transformer(tf.keras.models.Model):
	def __init__(self, n, h, inp_vocab_size, targ_vocab_size, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
		# TODO: Update document
		super(Transformer, self).__init__()
		self.encoder = Encoder(n, h, inp_vocab_size, d_model, d_ff, activation, dropout_rate, eps)
		self.decoder = Decoder(n, h, targ_vocab_size, d_model, d_ff, activation, dropout_rate, eps)
		self.ffn = Dense(targ_vocab_size)

	def call(self, encoder_in, decoder_in, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask):
		# TODO: Update document
		encoder_out = self.encoder(encoder_in, is_train, encoder_padding_mask)
		decoder_out, attention_weights = self.decoder(decoder_in, encoder_out, is_train, decoder_look_ahead_mask, decoder_padding_mask)
		return self.ffn(decoder_out)