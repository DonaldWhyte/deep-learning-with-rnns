import datetime
import logging
import tensorflow as tf
from tensorflow.contrib import layers
# `rnn` module temporarily in contrib. It's moving back to code in TF 1.1.
from tensorflow.contrib import rnn
from typing import Callable, List, Optional

from text import ALPHABET_SIZE, print_learning_learned_comparison

_logger = logging.getLogger(__name__)


# TODO: add dropout to this model


class RNNTextModel:

    def __init__(self,
                 sequence_length: int,
                 gru_internal_size: int,
                 num_hidden_layers: int,
                 stats_log_dir: str):
        """TODO

        TODO: document all inputs
        """
        self._sequence_length = sequence_length

        # Define RNN inputs.
        self._inputs = {
            'learning_rate': tf.placeholder(tf.float32, name='learning_rate'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),
            # Dimensions: [ batch_size, sequence_length ]
            'X': tf.placeholder(tf.uint8, [None, None], name='X')
            # Dimensions: [ batch_size, sequence_length, ALPHABET_SIZE ]
            'Xo': tf.one_hot(self._inputs['X'], ALPHABET_SIZE, 1.0, 0.0)
        }

        # Define expected RNN outputs. This is used for training.
        # This is the same sequence as the input sequence, but shifted by 1
        # since we are trying to predict the next character.
        self.expected_outputs = {
            # Dimensions: [ batch_size, sequence_length ]
            'Y':  = tf.placeholder(tf.uint8, [None, None], name='Y_exp')
            # Dimensions: [ batch_size, sequence_length, ALPHABET_SIZE ]
            'Yo':  = tf.one_hot(self._out, ALPHABET_SIZE, 1.0, 0.0)
        }

        # Define internal/hidden RNN layers. The RNN is composed of a certain
        # number of hidden layers, where each node is `GruCell` that uses
        # `gru_internal_size` as the internal state size of a single cell. A
        # higher `gru_internal_size` means more complex state can be stored in
        # a single cell.
        self._cells = [
            rnn.GRUCell(gru_internal_size) for _ in range(num_hidden_layers)]
        self._multicell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # When using state_is_tuple=True, you must use multicell.zero_state
        # to create a tuple of  placeholders for the input states (one state
        # per layer).
        #
        # When executed using session.run(self._zero_state), this also returns
        # the correctly shaped initial zero state to use when starting your
        # training loop.
        self._zero_state = multicell.zero_state(batch_size, dtype=tf.float32)

        # Using `dynamic_rnn` means Tensorflow "performs fully dynamic
        # unrolling" of the network. This is faster than compiling the full
        # graph at initialisation time.
        #
        # Note that compiling the full grapgh at train time isn’t that big of
        # an issue for training, because we only need to build the graph once.
        # It could be a big issue, however, if we need to build the graph
        # multiple times at test time. And remember, this training loop does
        # occassionally process inputs via test time, through the occassional
        # reports it outputs.
        #
        # Yr: [ batch_size, sequence_length, gru_internal_size ]
        # H:  [ batch_size, gru_internal_size * num_hidden_layers ]
        # H is the last state in the sequence.
        Yr, H = tf.nn.dynamic_rnn(
            multicell,
            Xo,
            dtype=tf.float32,
            initial_state=self._zero_state)

        # Do this just to give H a identifiable name
        H = tf.identity(H, name='H')

        # Softmax layer implementation:
        # Flatten the first two dimensions of the output. This performs the
        # following transformation:
        #
        # [ batch_size, sequence_length, ALPHABET_SIZE ]
        #     => [ batch_size x sequence_length, ALPHABET_SIZE ]
        Yflat = tf.reshape(Yr, [-1, gru_internal_size])

        # After this transformation, apply softmax readout layer. This way, the
        # weights and biases are shared across unrolled time steps. From the
        # readout point of view, a value coming from a cell or a minibatch is
        # the same thing.
        Ylogits = layers.linear(Yflat, ALPHABET_SIZE)                      # [ batch_size x sequence_length, ALPHABET_SIZE ]
        Yflat_ = tf.reshape(Yo_, [-1, ALPHABET_SIZE])                      # [ batch_size x sequence_length, ALPHABET_SIZE ]
        self._loss = tf.nn.softmax_cross_entropy_with_logits(          # [ batch_size x sequence_length ]
            logits=Ylogits, labels=Yflat_)
        self._loss = tf.reshape(                                       # [ batch_size, sequence_length ]
            self._loss, [self._inputs['batch_size'], -1])
        Yo = tf.nn.softmax(Ylogits, name='Yo')                         # [ batch_size x sequence_length, ALPHABET_SIZE ]
        Y = tf.argmax(Yo, 1)                                           # [ batch_size x sequence_length ]
        Y = tf.reshape(Y, [self._inputs['batch_size'], -1], name='Y')  # [ batch_size, sequence_length ]

        self._train_step = tf.train.AdamOptimizer().minimize(self._loss)

        self._build_statistics(stats_log_dir)
        self._initialise_tf_session()

    def _build_statistics(self, stats_log_dir: str):
        sequence_loss = tf.reduce_mean(loss, 1)
        batch_loss = tf.reduce_mean(sequence_loss)
        batch_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
        loss_summary = tf.summary.scalar('batch_loss', batch_loss)
        acc_summary = tf.summary.scalar('batch_accuracy', batch_accuracy)

        self._summaries = tf.summary.merge([loss_summary, acc_summary])
        self._stats_summary_writer = tf.summary.FileWriter(
            os.path.join(stats_log_dir, f'{timestamp}-training'))

        self. = tf.summary.FileWriter(
            os.path.join(stats_log_dir, f'{timestamp}-validation'))

    def _initialise_tf_session(self):
        initalised_vars = tf.global_variables_initializer()
        self._session = tf.Session()
        self._session.run(initalised_vars)
        self._initialised = True

    def run_training_loop(self,
                          training_text: List[int],
                          num_epochs: int,
                          learning_rate: float,
                          display_every_n_batches: int,
                          batch_size: int,
                          checkpoint_dir: str,
                          on_step_complete: Callable[[int], None],
                          should_stop: Callable[[int], bool]):
        """TODO

        TODO: document all args
        """
        # We display information on the currently trained model's accuracy
        # after training every N batches. Each batch corresponds to a single
        # step of the training loop. Each batch processes a number of sequences
        # which are X characters in length. Therefore, a step's size is defined
        # as the number of input sequences processed in a batch multiplied by
        # _length_ of each input sequence.
        step_size = batch_size * self._sequence_length
        display_every_n_batches = display_every_n_batches * step_size

        # Create direcftory that stores checkpoints of the trained model.
        # Training models can take a very long time (several hours or days), so
        # storing intermediate results on disk prevents us losing all progress
        # if the process crashes.
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=1)

        # Initial zero input state (a tuple).
        input_state = self._session.run(self._zero_state)

        batch_sequencer = txt.rnn_minibatch_sequencer(
            training_text, batch_size, sequence_length, nb_epochs=num_epochs)
        step = 0
        for x, y_, epoch in batch_sequencer:
            # Train on one minibatch.
            feed_dict = {
                self._inputs['X']: x,
                self._inputs['Y_']: y_,
                self._inputs['learning_rate']: learning_rate,
                self._inputs['batch_size']: batch_size
            }

            # This is how you add the input state to feed dictionary when
            # `state_is_tuple=True`.
            #
            #`self._zero_state` is a tuple of the placeholders for the
            # `num_hidden_layers` input states of our multi-layer RNN cell.
            # Those placeholders must be used as keys in `feed_dict`.
            #
            # `input_state` is a tuple holding the actual values of the input
            # states (one per layer). Iterate on the input state placeholders
            # and use them as keys in the dictionary to add actual input state
            # values.
            for i, v in enumerate(self._zero_state):
                feed_dict[v] = input_state[i]
            # This is the call that actuall runs the training batch.
            _, y, output_state, summary = self._session.run(
                [self._train_step, Y, H, self._summaries],
                feed_dict=feed_dict)

            # Save training data for Tensorboard
            self._summary_writer.add_summary(summary, step)

            # Display a visual validation of progress (every 50 batches)
            if step % display_every_n_batches == 0:
                # We don't use dropout for running the validation!
                feed_dict = {X: x, Y_: y_, batchsize: batch_size}
                for i, v in enumerate(self._zero_state):
                    feed_dict[v] = input_state[i]
                y, l, bl, acc = self._session.run(
                    [
                        self._expected_outputs['Y'],
                        self._sequence_loss,
                        self._batch_loss,
                        self._batch_accuracy
                    ],
                    feed_dict=feed_dict)
                print_learning_learned_comparison(
                    x[:5],
                    y,
                    l,
                    file_index,
                    bl,
                    acc,
                    epoch_size,
                    step,
                    epoch)

            # Save a checkpoint (every 500 batches)
            if step // 10 % display_every_n_batches == 0:
                saver.save(
                    self._session,
                    f'checkpoints/rnn_train_{timestamp}',
                    global_step=step)

            on_step_complete(step)
            if should_stop(step):
                break

            # Loop state around
            input_state = output_state
            step += step_size
