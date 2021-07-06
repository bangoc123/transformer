import os
from argparse import ArgumentParser
import tensorflow as tf
import logging
from data import NMTDataset
from transformer.model import Transformer
from transformer.optimizer import CustomLearningRate
from transformer.loss import loss_function
from trainer import Trainer
import io
from constant import *
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--test-path", default='{}/data/mock/test.en'.format(home_dir), type=str)
    parser.add_argument("--input-lang", default='en', type=str)
    parser.add_argument("--target-lang", default='vi', type=str)
    parser.add_argument("--input-path", default='{}/data/train/train.en'.format(home_dir), type=str)
    parser.add_argument("--target-path", default='{}/data/train/train.vi'.format(home_dir), type=str)
    parser.add_argument("--vocab-folder", default='{}/saved_vocab/transformer/'.format(home_dir), type=str)
    parser.add_argument("--checkpoint-folder", default='{}/checkpoints/'.format(home_dir), type=str)
    parser.add_argument("--buffer-size", default=64, type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--max-length", default=40, type=int)
    parser.add_argument("--num-examples", default=1000000, type=int)
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--n", default=6, type=int)
    parser.add_argument("--h", default=8, type=int)
    parser.add_argument("--d-ff", default=2048, type=int)
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--dropout-rate", default=0.1, type=float)
    parser.add_argument("--eps", default=0.1, type=float)

    args = parser.parse_args()
    print('---------------------Welcome to ProtonX Transformer-------------------')
    print('Github: bangoc123')
    print('Email: protonxai@gmail.com')
    print('---------------------------------------------------------------------')
    print('Predict using Transformer for text path: {}'.format(args.test_path))
    print('===========================')


    # Loading Tokenizer

    print('=============Loading Tokenizer================')
    print('Begin...')
    
    nmtdataset = NMTDataset(args.input_lang, args.target_lang, args.vocab_folder)
    inp_tokenizer, targ_tokenizer = nmtdataset.inp_tokenizer, nmtdataset.targ_tokenizer
    
    print('Done!!!')

    # Preprocessing sentences

    inp_lines = io.open(args.test_path, encoding=UTF_8).read().strip().split('\n')
    inp_lines = [nmtdataset.preprocess_sentence(inp, args.max_length) for inp in inp_lines]

    sentences = inp_tokenizer.texts_to_sequences(inp_lines)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(sentences, padding='post', maxlen=args.max_length)
    encoder_input = tf.convert_to_tensor(tensor, dtype=tf.int64)


    start, end = targ_tokenizer.word_index[BOS], targ_tokenizer.word_index[EOS]

    decoder_input = tf.convert_to_tensor([start], dtype=tf.int64)
    decoder_input = tf.expand_dims(decoder_input, 0)


    # Create custom Optimizer
    lrate = CustomLearningRate(args.d_model)
    
    optimizer = tf.keras.optimizers.Adam(lrate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    inp_vocab_size = len(inp_tokenizer.word_counts) + 1
    targ_vocab_size = len(targ_tokenizer.word_counts) + 1

    # Set checkpoint

    checkpoint_folder = args.checkpoint_folder

    print(args.n, args.h, inp_vocab_size, targ_vocab_size, args.d_model, args.d_ff, args.activation, args.dropout_rate, args.eps)
    # Initializing model
    transformer = Transformer(  
        args.n, 
        args.h, 
        inp_vocab_size, 
        targ_vocab_size, 
        args.d_model, 
        args.d_ff, 
        args.activation,
        args.dropout_rate,
        args.eps

    )

    trainer = Trainer(transformer, optimizer, args.epochs, checkpoint_folder)

    result = trainer.predict(encoder_input, decoder_input, False, args.max_length, end)
    
    final = targ_tokenizer.sequences_to_texts(result.numpy().tolist())
    print('---------> result: ', " ".join(final[0].split()[1:]))



    