
from transformer.layers.encoder import Encoder
from transformer.layers.decoder import Decoder
from tensorflow.keras.layers import Dense
from transformer.layers.generate_mask import generate_mask
import tensorflow as tf

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

	def call(self, input, training):
		# TODO: Update document
		encoder_in, decoder_in, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = input
		encoder_out = self.encoder(encoder_in, is_train, encoder_padding_mask)
		decoder_out, attention_weights = self.decoder(decoder_in, encoder_out, is_train, decoder_look_ahead_mask, decoder_padding_mask)
		return self.ffn(decoder_out)

	def compile(self, optimizer, loss_fn):
		super(Transformer, self).compile()
		self.optimizer = optimizer
		self.loss_fn = self.loss_function

	
	def loss_function(self, real, pred):
		mask = tf.math.logical_not(real == 0)
		loss = loss_object(real, pred)

		mask = tf.cast(mask, dtype=loss.dtype)
		loss = loss * mask

		return tf.reduce_sum(loss) / tf.reduce_sum(mask)

	def train_step(self, data):
		# TODO: Update document
		inp, tar = data
		tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
		encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask = generate_mask(inp, tar_inp)
		in_data = (inp, tar_inp, True, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)
		
		
		with tf.GradientTape() as tape:
			preds = self(in_data, True)
			d_loss = self.loss_fn(tar_real, preds)

		# Compute gradients
		grads = tape.gradient(d_loss, self.trainable_variables)

		# Update weights
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

		# Compute metrics
		train_loss.update_state(d_loss)
		train_accuracy.update_state(cal_acc(tar_real, preds))

		return {"loss": train_loss.result(), "acc": train_accuracy.result()}

	@property
	def metrics(self):
		# TODO: Update document
		# Reset state at the start of each epoch 
		return [train_loss, train_accuracy]
