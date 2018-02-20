from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np
from utils import *
from tensorflow.python.layers.core import Dense


class Seq2Seq:
    def __init__(self, rnn_size, n_layers, X_word2idx, encoder_embedding_dim, Y_word2idx, decoder_embedding_dim, batch,
                 sess=tf.Session(), grad_clip=5.0):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.X_word2idx = X_word2idx
        self.encoder_embedding_dim = encoder_embedding_dim
        self.Y_word2idx = Y_word2idx
        self.decoder_embedding_dim = decoder_embedding_dim
        self.sess = sess
        self.register_symbols()
        self.build_graph()
        self.batch = batch
        self.max_X_len = 0
        self.max_Y_len = 0
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_decoder_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.placeholder(tf.int32, [])
    # end method add_input_layer


    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method lstm_cell


    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.X_word2idx), self.encoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))            
        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]), 
            inputs = tf.nn.embedding_lookup(encoder_embedding, self.X),
            sequence_length = self.X_seq_len,
            dtype = tf.float32)
        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(self.n_layers))
    # end method add_encoder_layer
    

    def _attention(self, reuse=False):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units = self.rnn_size, 
            memory = self.encoder_out,
            memory_sequence_length = self.X_seq_len)
        
        return tf.contrib.seq2seq.AttentionWrapper(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse) for _ in range(self.n_layers)]),
            attention_mechanism = attention_mechanism,
            attention_layer_size = self.rnn_size)
    # end method add_attention


    def processed_decoder_input(self):
        main = tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1]) # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._y_go), main], 1)
        return decoder_input
    # end method add_decoder_layer


    def add_decoder_layer(self):
        with tf.variable_scope('decode'):
            decoder_embedding = tf.get_variable('decoder_embedding', [len(self.Y_word2idx), self.decoder_embedding_dim],
                                                 tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            decoder_cell = self._attention()
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
                sequence_length = self.Y_seq_len,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = training_helper,
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state),
                output_layer = Dense(len(self.Y_word2idx)))
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            decoder_cell = self._attention(reuse=True)
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = tf.get_variable('decoder_embedding'),
                start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
                end_token = self._y_eos)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = predicting_helper,
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state),
                output_layer = Dense(len(self.Y_word2idx), _reuse=True))
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
            self.predicting_ids = predicting_decoder_output.sample_id
    # end method add_decoder_layer


    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)

        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
    # end method add_backward_path


    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens
    # end method pad_sentence_batch


    def next_batch(self, X, Y, batch_size):
        for i in range(0, len(X) - len(X) % batch_size, batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            # padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            # padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(Y_batch, self._y_pad)
            yield (np.array(X_batch),
                   np.array(Y_batch),
                   [len(X_batch[0]) for i in range(len(X_batch))],
                   [len(Y_batch[0]) for i in range(len(Y_batch))])
    # end method next_batch


    def fit(self, X_train, Y_train, val_data, n_epoch=60, display_step=50, batch_size=128):
        X_test, Y_test = val_data
        X_test_batch, Y_test_batch, X_test_batch_lens, Y_test_batch_lens = next(
        self.next_batch(X_test, Y_test, batch_size))

        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                self.next_batch(X_train, Y_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X: X_train_batch,
                                                                     self.Y: Y_train_batch,
                                                                     self.X_seq_len: X_train_batch_lens,
                                                                     self.Y_seq_len: Y_train_batch_lens,
                                                                     self.batch_size: batch_size})
                if local_step % display_step == 0:
                    val_loss = self.sess.run(self.loss, {self.X: X_test_batch,
                                                         self.Y: Y_test_batch,
                                                         self.X_seq_len: X_test_batch_lens,
                                                         self.Y_seq_len: Y_test_batch_lens,
                                                         self.batch_size: batch_size})
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | test_loss: %.3f"
                        % (epoch, n_epoch, local_step, len(X_train)//batch_size, loss, val_loss))
    # end method fit


    def infer(self, input_word, X_idx2word, Y_idx2word, batch_size=128):        
        input_indices = input_word
        out_indices = self.sess.run(self.predicting_ids, {
            self.X: [input_indices] * batch_size,
            self.X_seq_len: [len(input_indices)] * batch_size,
            self.batch_size: batch_size})[0]
        
        # print('\nSource')
        # print('Word: {}'.format([i for i in input_indices]))
        # print('IN: {}'.format(' '.join([X_idx2word[i] for i in input_indices])))
        #
        # print('\nTarget')
        # print('Word: {}'.format([i for i in out_indices]))
        # print('OUT: {}'.format(' '.join([Y_idx2word[i] for i in out_indices])))
        return [Y_idx2word[i] for i in out_indices]
    # end method infer


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = 0
        self._x_unk = self.X_word2idx['<UNK>']

        self._y_go = self.Y_word2idx['<GO>']
        self._y_eos = self.Y_word2idx['<EOS>']
        self._y_pad = 0
        self._y_unk = self.Y_word2idx['<UNK>']
    # end method add_symbols




    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {}
        batch_ph = [np.reshape(X.T[t], (X.shape[0], 1)) for t in range(self.batch)]
        target_ph = [np.reshape(Y.T[t], (Y.shape[0], 1)) for t in range(self.batch)]

        feed_dict = {self.X: np.reshape(batch_ph, (self.batch, X.shape[0]))}
        feed_dict.update({self.Y: np.reshape(target_ph, (self.batch, Y.shape[0]))})

        feed_dict.update({self.X_seq_len: [get_eos_pos(t, self.max_X_len) for t in batch_ph]})
        feed_dict.update({self.Y_seq_len: [get_eos_pos(t, self.max_Y_len) for t in target_ph]})

        # feed_dict[self.keep_prob] = keep_prob  # dropout prob
        feed_dict[self.batch_size] = len(feed_dict[self.X_seq_len])  # dropout prob

        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen, merged_summary_op):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feedte
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, summary = sess.run([self.train_op, self.loss], feed_dict)

        # _, summary = sess.run([self.train_op, merged_summary_op], feed_dict)
        # tf.summary.scalar("loss_function", loss_v)

        return summary

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    # finally the train function that
    #  runs the train_op in a session
    #   evaluates on valid set periodically
    #    prints statistics
    def train(self, train_set, valid_set, max_X_len, max_Y_len, epochs=10,sess=None):

        # we need to save the model periodically
        saver = tf.train.Saver()

        self.max_X_len = max_X_len
        self.max_Y_len = max_Y_len
        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter('/tf_logs/', graph=tf.get_default_graph())

        prt('Training started </log>\n')
        # tf.summary.scalar("bleu", self.bleu)
        merged_summary_op = tf.summary.merge_all()

        # run M epochs
        for i in range(epochs):
            try:
                summary = self.train_batch(sess, train_set, merged_summary_op)
                summary_writer.add_summary(summary, i)

                if i and i % (1) == 0:  # TODO : make this tunable by the user
                    # save model to disk
                    # saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 1)  # TODO : and this
                    # print stats
                    prt('\nModel saved to disk at iteration #{}'.format(i))
                    prt('val   loss : {0:.6f}'.format(val_loss))
                    # sys.stdout.flush()
            except KeyboardInterrupt:  # this will most definitely happen, so handle it
                prt('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess
        self.session = sess
        return sess
# end class