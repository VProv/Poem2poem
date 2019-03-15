import pickle
import numpy as np
import tensorflow as tf
import keras.layers as L
from keras import backend as K
from utils import infer_length, infer_mask
from translation_model_interface import ITranslationModel


class AttentionLayer:
    
    def __init__(self, name, hid_size, activ=tf.tanh,):
        """ A layer that computes additive attention response and weights """
        self.name = name
        self.hid_size = hid_size # attention layer hidden units
        self.activ = activ       # attention layer hidden nonlinearity

        with tf.variable_scope(name):
            # YOUR CODE - create layer variables
            #<YOUR CODE>
            self.linear_e = L.Dense(hid_size)
            self.linear_d = L.Dense(hid_size)
            self.linear_out = L.Dense(1)

    def __call__(self, enc, dec, inp_mask):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param inp_mask: mask on enc activatons (0 after first eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """
        with tf.variable_scope(self.name):
            
            # Compute logits
            #<...>
            logits_seq = self.linear_out(self.activ(self.linear_e(enc) + \
                                                    self.linear_d(dec)[:, tf.newaxis, :]))
            logits_seq = tf.squeeze(logits_seq, axis = -1)
            
            # Apply mask - if mask is 0, logits should be -inf or -1e9
            # You may need tf.where
            #<...>
            
            logits_seq = tf.where(inp_mask, logits_seq, tf.fill(tf.shape(logits_seq),
                                                                -np.inf))
            
            # Compute attention probabilities (softmax)
            probs = tf.nn.softmax(logits_seq) # <...>
            
            # Compute attention response using enc and probs
            attn = tf.reduce_sum(probs[..., tf.newaxis] * enc, axis = 1) # <...>
            
            return attn, probs
        
class AttentiveModel(ITranslationModel):
    
    def __init__(self, sess, filename, name = None, inp_voc = None, out_voc = None,
                 emb_size = None, hid_size = None):
        
        self.sess = sess
        
        if filename is None:
            self.initialize(name, inp_voc, out_voc,
                            emb_size, hid_size) #, attn_size)
        else:
            self.load(filename)
    
    
    def initialize(self, name, inp_voc, out_voc,
                   emb_size, hid_size): #, attn_size):
        
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.emb_size = emb_size
        self.hid_size = hid_size
        #self.attn_size = attn_size

        with tf.variable_scope(name):
            
            # YOUR CODE - define model layers
            
            # <...>
            self.emb_inp = L.Embedding(len(inp_voc), emb_size)
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            self.enc_lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_size,
                                                                 forget_bias=1.0,
                                                                 state_is_tuple = False)
            self.enc_lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_size,
                                                                 forget_bias=1.0,
                                                                 state_is_tuple = False)
            #self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size)

            self.dec_start = L.Dense(hid_size)
            self.dec0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.dense = L.Dense(hid_size)
            self.activ = tf.tanh
            self.logits = L.Dense(len(out_voc))
            
            self.attention = AttentionLayer(name = 'attention',
                                            #enc_size = None, # FIXME: Unused
                                            #dec_size = None, # FIXME: Unused
                                            #hid_size = attn_size)
                                            hid_size = 2 * self.hid_size)
            
            # END OF YOUR CODE
            
            # prepare to translate_lines
            self.inp = tf.placeholder('int32', [None, None])
            self.initial_state = self.prev_state = self.encode(self.inp)
            self.prev_tokens = tf.placeholder('int32', [None])
            self.next_state, self.next_logits = self.decode(self.prev_state, self.prev_tokens)
            self.next_softmax = tf.nn.softmax(self.next_logits)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        
        # Call to 'K.get_session()' runs variable initializes for
        # all variables including ones initialized using
        # 'tf.global_variables_initializer()' (at least for Keras
        # 2.0.5) thus it have to be called once here or model weights
        # will be rewritten after training e.g. when 'get_weights' is
        # called.
        K.get_session()

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        
        # encode input sequence, create initial decoder states
        # <YOUR CODE>
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        inp_mask = infer_mask(inp, self.inp_voc.eos_ix, dtype = tf.bool)
        
        inp_emb = self.emb_inp(inp)
        with tf.variable_scope('enc0'):
            #enc_seq, enc_last = tf.nn.dynamic_rnn(self.enc0,
            #                                      inp_emb,
            #                                      sequence_length = inp_lengths,
            #                                      dtype = inp_emb.dtype)
            ((enc_seq_fw,
              enc_seq_bw),
             (enc_last_fw,
              enc_last_bw)) = tf.nn.bidirectional_dynamic_rnn(self.enc_lstm_fw_cell,
                                                              self.enc_lstm_bw_cell,
                                                              inp_emb,
                                                              sequence_length = inp_lengths,
                                                              dtype = inp_emb.dtype)
        enc_seq = tf.concat((enc_seq_fw, enc_seq_bw), axis = -1)
        dec_start = self.dec_start(enc_last_fw)
        
        # apply attention layer from initial decoder hidden state
        #first_attn_probas = <...>
        _, first_attn_probas = self.attention(enc_seq, dec_start, inp_mask)
        
        # Build first state: include
        # * initial states for decoder recurrent layers
        # * encoder sequence and encoder attn mask (for attention)
        # * make sure that last state item is attention probabilities tensor
        
        #first_state = [<...>, first_attn_probas]
        first_state = [dec_start, enc_seq, inp_mask, first_attn_probas]
        return first_state

    def decode(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        # Unpack your state: you will get tensors in the same order
        # that you've packed in encode
        #[<...>, prev_attn_probas] = prev_state
        [prev_dec, enc_seq, inp_mask, prev_attn_probas] = prev_state
        
        
        # Perform decoder step
        # * predict next attn response and attn probas given previous decoder state
        # * use prev token embedding and attn response to update decoder states
        # * (concatenate and feed into decoder cell)
        # * predict logits
        
        # <APPLY_ATTENTION>
        next_attn_response, next_attn_probas = self.attention(enc_seq, prev_dec, inp_mask)

        # <YOUR CODE>
        prev_emb = self.emb_out(prev_tokens[:,None])[:,0]
        dec_inputs = tf.concat([prev_emb, next_attn_response], axis = 1)
        with tf.variable_scope('dec0'):
            new_dec_out, new_dec_state = self.dec0(dec_inputs, prev_dec)
        output_logits = self.logits(self.activ(self.dense(new_dec_out)))
        #output_logits = self.logits(self.activ(new_dec_out))
        
        # Pack new state:
        # * replace previous decoder state with next one
        # * copy encoder sequence and mask from prev_state
        # * append new attention probas
        #next_state = [<...>, next_attn_probas]
        next_state = [new_dec_state, enc_seq, inp_mask, next_attn_probas]
        return next_state, output_logits

    
    def compute_logits(self, inp, out, **flags):
        
        batch_size = tf.shape(inp)[0]

        # Encode inp, get initial state
        first_state = self.encode(inp) # <YOUR CODE HERE>

        # initial logits: always predict BOS
        first_logits = tf.log(tf.one_hot(tf.fill([batch_size], self.out_voc.bos_ix),
                                         len(self.out_voc)) + 1e-30)

        # Decode step
        def step(prev_state, y_prev):
            # Given previous state, obtain next state and next token logits
            # <YOUR CODE>
            next_dec_state, next_logits = self.decode(prev_state, y_prev)
            return next_dec_state, next_logits # <...>

        # You can now use tf.scan to run step several times.
        # use tf.transpose(out) as elems (to process one time-step at a time)
        # docs: https://www.tensorflow.org/api_docs/python/tf/scan

        # <YOUR CODE>

        out = tf.scan(lambda a, y: step(a[0], y),
                      elems = tf.transpose(out)[:-1],
                      initializer = (first_state, first_logits))


        # FIXME remove?
        #self.sess.run(tf.initialize_all_variables())

        logits_seq = out[1] # <YOUR CODE>

        # prepend first_logits to logits_seq
        logits_seq = tf.concat((first_logits[tf.newaxis], logits_seq), axis = 0) #<...>

        # Make sure you convert logits_seq from
        # [time, batch, voc_size] to [batch, time, voc_size]
        logits_seq = tf.transpose(logits_seq, perm = [1, 0, 2]) #<...>

        return logits_seq

    def compute_loss(self, inp, out, **flags):
        
        mask = infer_mask(out, out_voc.eos_ix)    
        logits_seq = self.compute_logits(inp, out, **flags)

        # Compute loss as per instructions above
        # <YOUR CODE>

        prob_seq = tf.nn.softmax(logits_seq)
        out_one_hot = tf.one_hot(out, len(self.out_voc))

        prob_seq_masked = tf.boolean_mask(prob_seq, mask)
        out_one_hot_masked = tf.boolean_mask(out_one_hot, mask)
        prob_seq_out = tf.boolean_mask(prob_seq_masked, out_one_hot_masked)
        loss = tf.reduce_mean(-tf.log(prob_seq_out))

        return loss
    
    
    def make_initial_state(self, inp_lines):
        return self.sess.run(self.initial_state, {self.inp: self.inp_voc.to_matrix(inp_lines)})
    
    def get_next_state_and_logits(self, state, outputs):
        return self.sess.run([self.next_state, self.next_logits],
                        {**dict(zip(self.prev_state, state)),
                         self.prev_tokens: [out_i[-1] for out_i in outputs]})
                         
    def get_output_vocabulary(self):
        return self.out_voc
    
    
    def translate_lines(self, inp_lines, max_len=100):
        """
        Translates a list of lines by greedily selecting most likely next token at each step
        :returns: a list of output lines, a sequence of model states at each step
        """
        state = self.make_initial_state(inp_lines)
        outputs = [[self.out_voc.bos_ix] for _ in range(len(inp_lines))]
        all_states = [state]
        finished = [False] * len(inp_lines)

        for t in range(max_len):
            state, logits = self.get_next_state_and_logits(state, outputs)
            next_tokens = np.argmax(logits, axis=-1)
            all_states.append(state)
            for i in range(len(next_tokens)):
                outputs[i].append(next_tokens[i])
                finished[i] |= next_tokens[i] == self.out_voc.eos_ix
        return self.out_voc.to_lines(outputs), all_states
    
    def dump(self, filename):
        
        values = {'name': self.name,
                  'inp_voc': self.inp_voc,
                  'out_voc': self.out_voc,
                  'emb_size': self.emb_size,
                  'hid_size': self.hid_size,
                  #'attn_size': self.attn_size,
                  'emb_inp_weights': self.emb_inp.get_weights(),
                  'emb_out_weights': self.emb_out.get_weights(),
                  #'enc0_weights': self.enc0.get_weights(),
                  'enc_lstm_fw_cell_weights': self.enc_lstm_fw_cell.get_weights(),
                  'enc_lstm_bw_cell_weights': self.enc_lstm_bw_cell.get_weights(),
                  'dec0_weights': self.dec0.get_weights(),
                  'dec_start_weights': self.dec_start.get_weights(),
                  'dense_weights': self.dense.get_weights(),
                  'logits_weights': self.logits.get_weights(),
                  'attn__linear_e_weights': self.attention.linear_e.get_weights(),
                  'attn__linear_d_weights': self.attention.linear_d.get_weights(),
                  'attn__linear_out_weights': self.attention.linear_out.get_weights()}
        pickle.dump(values, open(filename, 'wb'))
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            values = pickle.load(f)
        self.initialize(values['name'], values['inp_voc'], values['out_voc'],
                        values['emb_size'], values['hid_size']) #, values['attn_size'])
        self.emb_inp.set_weights(values['emb_inp_weights'])
        self.emb_out.set_weights(values['emb_out_weights'])
        #self.enc0.set_weights(values['enc0_weights'])
        self.enc_lstm_fw_cell.set_weights(values['enc_lstm_fw_cell_weights'])
        self.enc_lstm_bw_cell.set_weights(values['enc_lstm_bw_cell_weights'])
        self.dec0.set_weights(values['dec0_weights'])
        self.dec_start.set_weights(values['dec_start_weights'])
        self.dense.set_weights(values['dense_weights'])
        self.logits.set_weights(values['logits_weights'])
        self.attention.linear_e.set_weights(values['attn__linear_e_weights'])
        self.attention.linear_d.set_weights(values['attn__linear_d_weights'])
        self.attention.linear_out.set_weights(values['attn__linear_out_weights'])
