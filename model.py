# from model import CVAE
# from utils import *
import numpy as np
import os
import tensorflow as tf
import time
import argparse
import collections
import tensorflow_addons as tfa


# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
# args = parser.parse_args()
# print (args)

class CVAE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.batch_size = 128
        self.lr = tf.Variable(0.0001, trainable=False)
        self.unit_size = 512
        self.n_rnn_layers = 3

        self.createNetwork()

    def createNetwork(self):
        encoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layers)]
        print(encoded_rnn_size)

        with tf.compat.v1.variable_scope('rnn'):
            encoder_cell = []
            for i in encoded_rnn_size[:]:
                encoder_cell.append(tf.compat.v1.nn.rnn_cell.LSTMCell(i))
            self.encoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                encoder_cell)

        self.weights = {}
        self.biases = {}
        # print("ho")
        print(self.encoder_cell)

        self.weights['softmax'] = tf.compat.v1.get_variable("softmaxw", initializer=tf.compat.v1.random_uniform(
            shape=[encoded_rnn_size[-1], self.vocab_size], minval=-0.1, maxval=0.1))

        self.biases['softmax'] = tf.compat.v1.get_variable(
            "softmaxb", initializer=tf.compat.v1.random_uniform(shape=[self.vocab_size]))
        self.embedding_encode = tf.compat.v1.get_variable(name='encode_embedding', shape=[
                                                          self.unit_size, self.vocab_size], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        self.decoded, decoded_logits = self.rnn()

        weights = tf.sequence_mask(self.L, tf.shape(self.X)[1])
        weights = tf.cast(weights, tf.int32)
        weights = tf.cast(weights, tf.float32)
        self.reconstr_loss = tf.reduce_mean(tf.seq2seq.sequence_loss())

        # Loss
        self.loss = self.reconstr_loss
        #self.loss = self.reconstr_loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.opt = optimizer.minimize(self.loss)

        self.mol_pred = tf.argmax(self.decoded, axis=2)
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=None)
        # tf.train.start_queue_runners(sess=self.sess)
        print("Network Ready")


def load_data(char, vocab, seq_length=120):
    with open("smiles.csv") as f:
        lines = f.read().split('\n')[:-1]
    smiles = [l for l in lines if len(l) < seq_length-2]
    smiles_input = []
    smiles_output = []
    length = []
    for s in smiles:
        length.append(len(s)+1)
        s1 = ('X'+s).ljust(seq_length, 'E')
        s2 = s.ljust(seq_length, 'E')
        list1 = list(map(vocab.get, s1))
        list2 = list(map(vocab.get, s2))
        if None in list1 or None in list2:
            continue
        smiles_input.append(list1)
        smiles_output.append(list2)
    smiles_input = np.array(smiles_input)
    smiles_output = np.array(smiles_output)
    length = np.array(length)
    return smiles_input, smiles_output, length


def extract_vocab(seq_length=120):
    with open("smiles.csv") as f:
        lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l[0]) < seq_length-2]
    smiles = [l[0] for l in lines]
    # print(lines)
    # print(smiles)
    total_string = ''
    for s in smiles:
        total_string += s
    counter = collections.Counter(total_string)
    # print(counter)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    # print(count_pairs)
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))

    chars += ('E',)  # End of smiles
    chars += ('X',)  # Start of smiles
    vocab['E'] = len(chars)-2
    vocab['X'] = len(chars)-1
    # print(vocab)
    # print(chars)
    return chars, vocab


char, vocab = extract_vocab()
molecules_input, molecules_output, length = load_data(char, vocab)
print(molecules_input)
print(molecules_output)
print(length)
print('Number of data : ', len(molecules_input))
vocab_size = len(char)

if not os.path.isdir("./save"):
    os.mkdir("./save")

num_train_data = int(len(molecules_input)*0.75)
train_molecules_input = molecules_input[0:num_train_data]
test_molecules_input = molecules_input[num_train_data:-1]

train_molecules_output = molecules_output[0:num_train_data]
test_molecules_output = molecules_output[num_train_data:-1]

train_length = length[0:num_train_data]
test_length = length[num_train_data:-1]

# print(train_length)
# print(test_length)

model = CVAE(vocab_size)
