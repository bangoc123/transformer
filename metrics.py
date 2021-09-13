import collections
import numpy as np
import tensorflow as tf

from constant import BOS, EOS
from transformer.layers.generate_mask import generate_mask


class BleuScore:
    """
        We can evaluate a predicted sequence by comparing it with the label sequence.
        BLEU (Bilingual Evaluation Understudy) "https://aclanthology.org/P02-1040.pdf",
        though originally proposed for evaluating machine translation results,
        has been extensively used in measuring the quality of output sequences for different applications.
        In principle, for any n-grams in the predicted sequence, BLEU evaluates whether this n-grams appears
        in the label sequence.
    """

    def __init__(self, inp_tokenizer, targ_tokenizer, n_grams=3):
        self.inp_tokenizer = inp_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.start = targ_tokenizer.word_index[BOS]
        self.end = targ_tokenizer.word_index[EOS]
        self.n_grams = n_grams

    def remove_oov(self, sentence):
        return [i for i in sentence[0].split() if i not in [self.start, self.end]]

    def __call__(self, target, pred):
        target = self.targ_tokenizer.sequences_to_texts([target])
        pred = self.targ_tokenizer.sequences_to_texts([pred])

        target = self.remove_oov(target)
        pred = self.remove_oov(pred)

        pred_length = len(pred)
        target_length = len(target)

        if pred_length < self.n_grams:
            return 0
        else:
            score = np.exp(np.minimum(0, 1 - target_length / pred_length))
            for k in range(1, self.n_grams + 1):
                label_subs = collections.defaultdict(int)
                for i in range(target_length - k + 1):
                    label_subs[" ".join(target[i:i + k])] += 1

                num_matches = 0
                for i in range(pred_length - k + 1):
                    if label_subs[" ".join(pred[i:i + k])] > 0:
                        label_subs[" ".join(pred[i:i + k])] -= 1
                        num_matches += 1
                score *= np.power(num_matches / (pred_length - k + 1), np.power(0.5, k))
            return score


class Evaluate:
    def __init__(self, max_length, inp_tokenizer, targ_tokenizer, n_grams=3):
        self.max_length = max_length

        self.start_token = targ_tokenizer.word_index[BOS]
        self.end_token = targ_tokenizer.word_index[EOS]

        self.bleu = BleuScore(inp_tokenizer, targ_tokenizer, n_grams)

    def __call__(self, model, val_inps, val_tars):
        score = 0
        for inp, tar_inp in zip(val_inps, val_tars):
            encoder_input = tf.cast(tf.expand_dims(inp, axis=0), dtype=tf.int64)
            decoder_input = tf.reshape(tf.constant([self.start_token]), shape=(1, -1))
            decoder_input = tf.cast(decoder_input, dtype=tf.int64)
            for i in range(self.max_length):
                encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = generate_mask(encoder_input,
                                                                                                    decoder_input)

                preds = model(encoder_input, decoder_input, False, encoder_padding_mask,
                              decoder_look_ahead_mask, decoder_padding_mask)

                preds = preds[:, -1:, :]  # (batch_size, 1, vocab_size)

                predicted_id = tf.argmax(preds, axis=-1)

                decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

                # return the result if the predicted_id is equal to the end token
                if predicted_id == self.end_token:
                    break
            pred_sentence = decoder_input[0].numpy()
            target_sentence = tar_inp.numpy()
            score += self.bleu(target_sentence, pred_sentence)
        return score / len(val_tars)


if __name__ == '__main__':
    """
    python train.py --epochs=200 --input-lang en --target-lang vi --input-path="data/mock/train.en" --target-path="data/mock/train.vi" --use-bleu=True
    """
