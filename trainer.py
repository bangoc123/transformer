from transformer.layers.generate_mask import generate_mask
import tensorflow as tf

class Trainer:
	def __init__(self, model, optimizer, epochs, checkpoint_folder):
		self.model = model
		self.optimizer = optimizer
		self.epochs = epochs
		self.train_loss = tf.keras.metrics.Mean(name='train_loss')
		self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
		self.checkpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
		self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_folder, max_to_keep=3)

	def cal_acc(self, real, pred):
		accuracies = tf.equal(real, tf.argmax(pred, axis=2))

		mask = tf.math.logical_not(real == 0)
		accuracies = tf.math.logical_and(mask, accuracies)

		accuracies = tf.cast(accuracies, dtype=tf.float32)
		mask = tf.cast(mask, dtype=tf.float32)
		return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

	def loss_function(self, real, pred):
		cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss = cross_entropy(real, pred)

		mask = tf.cast(mask, dtype=loss.dtype)
		loss = loss * mask
		return tf.reduce_sum(loss) / tf.reduce_sum(mask)

	def train_step(self, inp, tar):
		# TODO: Update document
		tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
		encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask = generate_mask(inp, tar_inp)

		with tf.GradientTape() as tape:
			preds = self.model(inp, tar_inp, True, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)
			d_loss = self.loss_function(tar_real, preds)

		# Compute gradients
		grads = tape.gradient(d_loss, self.model.trainable_variables)

		# Update weights
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

		# Compute metrics
		self.train_loss.update_state(d_loss)
		self.train_accuracy.update_state(self.cal_acc(tar_real, preds))

		# return {"loss": self.train_loss.result(), "acc": self.train_accuracy.result()}

	def fit(self, data):
		print('=============Training Progress================')
		print('----------------Begin--------------------')
		# Loading checkpoint
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			print('Restored checkpoint manager !')

		for epoch in range(self.epochs):
			self.train_loss.reset_states()
			self.train_accuracy.reset_states()
			
			for (batch, (inp, tar)) in enumerate(data):
				self.train_step(inp, tar)

				if batch % 50 == 0:
					print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.3f} Accuracy {self.train_accuracy.result():.3f}')

				if (epoch + 1) % 5 == 0:
					saved_path = self.checkpoint_manager.save()
					print('Checkpoint was saved at {}'.format(saved_path))
		print('----------------Done--------------------')