import os
from argparse import ArgumentParser
import tensorflow as tf
import logging
from data import NMTDataset
from transformer.model import Transformer
from transformer.optimizer import CustomLearningRate
from transformer.loss import loss_function
from trainer import Trainer
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--input-lang", default='en', type=str, required=True)
    parser.add_argument("--target-lang", default='vi', type=str, required=True)
    parser.add_argument("--input-path", default='{}/data/train/train.en'.format(home_dir), type=str, required=True)
    parser.add_argument("--target-path", default='{}/data/train/train.vi'.format(home_dir), type=str, required=True)
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

    print('---------------------Welcome to ProtonX Transfomer-------------------')
    print('Github: bangoc123')
    print('Email: protonxai@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Transfomer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')


    nmtdataset = NMTDataset(args.input_lang, args.target_lang, args.vocab_folder)
    train_dataset, val_dataset = nmtdataset.build_dataset(args.input_path, args.target_path, args.buffer_size, args.batch_size, args.max_length, args.num_examples)

    inp_tokenizer, targ_tokenizer = nmtdataset.inp_tokenizer, nmtdataset.targ_tokenizer

    # Create custom Optimizer
    lrate = CustomLearningRate(args.d_model)
    
    optimizer = tf.keras.optimizers.Adam(lrate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    inp_vocab_size = len(inp_tokenizer.word_counts) + 1
    targ_vocab_size = len(targ_tokenizer.word_counts) + 1

    # Set checkpoint

    checkpoint_folder = args.checkpoint_folder

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

    # trainer = Trainer(transformer, optimizer, args.epochs, checkpoint_folder)

    # Training model
    # trainer.fit(train_dataset)
    
    # Saving model
    # transformer.save_weights(args.model_folder)


    # Using Transformer Fit
    transformer.compile(
        optimizer = optimizer,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    )

    # Creat Checkpoint: saves the model's weights every 2 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_folder, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch',
        monitor="val_acc")


    transformer.fit(
        train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[cp_callback],
    )

    checkpoint = tf.train.Checkpoint(model = transformer, optimizer = optimizer)
