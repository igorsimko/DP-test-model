from tensorflow.python.layers.core import Dense
from utils import *

tf.logging.set_verbosity(tf.logging.DEBUG)
setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


class Seq2Seq(object):
    def __init__(self, xseq_len, yseq_len,
                 xvocab_size, yvocab_size,
                 emb_dim, num_layers, ckpt_path, num_units, batch_size=32,
                 lr=0.0001, embedding=None, emb_size=1,
                 epochs=11, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.xvocab_size = xvocab_size
        self.yvocab_size = yvocab_size

        self.batch_size = batch_size
        self.embedding = embedding
        self.emb_size = emb_size

        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.num_units = num_units
        self.num_layers = num_layers
        self.model_name = model_name
        self.lr = lr

        self.merged_summary_op = None

        self.keep_prob = 0
        self.batch_ph = 0
        self.target_ph = 0
        self.batch_size_ph = 0

        self.Xseq_len_ph = 0
        self.Yseq_len_ph = 0

        self.source_seq_len = xseq_len
        self.go_token = 0
        self.eos_token = 2

        self.init_graph()

    def init_graph(self):
        # placeholders
        tf.reset_default_graph()
        prt("Building graph")

        # Different placeholders
        self.batch_ph = tf.placeholder(tf.int32, [None, None])
        self.target_ph = tf.placeholder(tf.int32, [None, None])

        self.Xseq_len_ph = tf.placeholder(tf.int32, [None])
        self.Yseq_len_ph = tf.placeholder(tf.int32, [None])

        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size_ph = tf.placeholder(tf.int32, [])
        self.encoder_embedding_ph = tf.placeholder(tf.float32, [None, self.emb_size])

        prt("Encoder start.")
        # ENCODER
        # encoder_embedding = tf.get_variable('encoder_embedding', [self.xvocab_size, self.emb_size],
        #                                     tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        #
        tf.summary.histogram('embeddings_var', self.encoder_embedding_ph)

        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)]),
            inputs=tf.nn.embedding_lookup(self.encoder_embedding_ph, self.batch_ph),
            sequence_length=self.Xseq_len_ph,
            dtype=tf.float32)
        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(self.num_layers))

        prt("Encoder done.")

        with tf.variable_scope(tf.get_variable_scope(), reuse=None) as scope:
            prt("Decoder start.")

            # decoder_embedding = tf.get_variable('decoder_embedding',
            #                                     [self.yvocab_size, self.emb_size],
            #                                     tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            decoder_cell = self.attention()
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(self.encoder_embedding_ph, self.processed_decoder_input()),
                sequence_length=self.Yseq_len_ph,
                time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=decoder_cell.zero_state(self.batch_size_ph, tf.float32).clone(
                    cell_state=self.encoder_state),
                output_layer=Dense(self.yvocab_size)
            )
            self.decode_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=tf.reduce_max(self.Yseq_len_ph))
            self.training_logits = self.decode_outputs.rnn_output
            prt("Decoder done.")

            scope.reuse_variables()

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            decoder_cell = self.attention(True)
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.encoder_embedding_ph,
                start_tokens=tf.tile(tf.constant([self.go_token], dtype=tf.int32), [self.batch_size_ph]),
                end_token=self.eos_token)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=predicting_helper,
                initial_state=decoder_cell.zero_state(self.batch_size_ph, tf.float32).clone(
                    cell_state=self.encoder_state),
                output_layer=Dense(self.yvocab_size, _reuse=True))
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=True,
                maximum_iterations=2 * tf.reduce_max(self.Xseq_len_ph))
            self.predicting_ids = predicting_decoder_output.sample_id

        # with tf.name_scope('accuracy'):
        #     correct_prediction = tf.equal(tf.argmax(self.training_logits, 1), tf.argmax(self.target_ph, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        #     tf.summary.scalar('train_accuracy', accuracy)

        # LOSS
        # self.loss = self._compute_loss(self.training_logits)
        prt("Backward pass start.")

        masks = tf.sequence_mask(self.Yseq_len_ph, tf.reduce_max(self.Yseq_len_ph), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                     targets=self.target_ph,
                                                     weights=masks)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
        prt("Backward pass done.")
        prt("Building graph done.")


    '''
        Training and Evaluation

    '''

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.target_ph

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.Yseq_len_ph, None, dtype=logits.dtype)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size_ph)
        return loss

    def processed_decoder_input(self):
        main = tf.strided_slice(self.target_ph, [0, 0], [self.batch_size, -1], [1, 1])  # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.go_token), main], 1)
        return decoder_input

    # ATTENTION
    def attention(self, reuse=False):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.num_units,
            memory=self.encoder_out,
            memory_sequence_length=self.Xseq_len_ph)

        wrapper = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse) for _ in range(self.num_layers)]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.num_units)

        return wrapper

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.num_units, initializer=tf.orthogonal_initializer(), reuse=reuse)


    # prediction
    def predict(self, sess, X, idx2word, embedding):
        out = sess.run(self.predicting_ids, {
            self.batch_ph: [X] * self.batch_size,
            self.Xseq_len_ph: [len(X)] * self.batch_size,
            self.batch_size_ph: self.batch_size,
            self.encoder_embedding_ph: embedding})[0]

        return [idx2word[i] for i in out]

    def next_batch(self, X, Y, batch_size):
        for i in range(0, len(X) - len(X) % batch_size, batch_size):
            X_batch = X[i: i + batch_size]
            Y_batch = Y[i: i + batch_size]
            # padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            # padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(Y_batch, self._y_pad)
            yield (np.array(X_batch),
                   np.array(Y_batch),
                   [get_eos_pos(i, self.xseq_len) for i in X_batch],

                   # [len(X_batch[0]) for i in range(len(X_batch))],
                   [len(Y_batch[0]) for i in range(len(Y_batch))])
                   #  [get_eos_pos(i, self.yseq_len) for i in Y_batch])

    # end method next_batch

    def fit(self, X_train, Y_train, val_data, log_dir, embedding, sess=None, display_step=50, batch_size=128):
        saver = tf.train.Saver()

        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        prt('Training started\n')
        tf.summary.scalar("loss", self.loss)

        self.merged_summary_op = tf.summary.merge_all()
        # embedding = tf.cast(embedding, tf.float32)
        good_loss = False
        for epoch in range(1, self.epochs + 1):
            if good_loss:
                break
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                    self.next_batch(X_train, Y_train, batch_size)):
                _, summary = sess.run([self.train_op, self.merged_summary_op], {self.batch_ph: X_train_batch,
                                                                self.target_ph: Y_train_batch,
                                                                self.Xseq_len_ph: X_train_batch_lens,
                                                                self.Yseq_len_ph: Y_train_batch_lens,
                                                                self.batch_size_ph: batch_size,
                                                                self.encoder_embedding_ph: embedding})
                if local_step % display_step == 0:
                    self.n_epoch = epoch
                    val_loss = sess.run(self.loss, {self.batch_ph: X_train_batch,
                                                    self.target_ph: Y_train_batch,
                                                    self.Xseq_len_ph: X_train_batch_lens,
                                                    self.Yseq_len_ph: Y_train_batch_lens,
                                                    self.batch_size_ph: batch_size,
                                                    self.encoder_embedding_ph: embedding})
                    prt("Epoch %d/%d |  test_loss: %.3f" % (epoch, self.epochs, val_loss))
                    save_path = saver.save(sess, self.ckpt_path + "model.ckpt")
                    if val_loss <= 0.001:
                        good_loss = True
                        break

                summary_writer.add_summary(summary, epoch)
        save_path = saver.save(sess, self.ckpt_path + "model.ckpt")
        prt("Model saved in path: %s" % save_path)
        return sess
