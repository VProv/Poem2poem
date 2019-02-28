import tensorflow as tf
import keras.layers as L
import numpy as np
from models.utils import infer_length, infer_mask
from models.utils import select_values_over_last_axis, compute_logits, compute_loss, compute_bleu

class AttentionLayer:
    def __init__(self, name, enc_size, dec_size, hid_size, activ=tf.tanh,):
        """ A layer that computes additive attention response and weights """
        self.name = name
        self.enc_size = enc_size # num units in encoder state
        self.dec_size = dec_size # num units in decoder state
        self.hid_size = hid_size # attention layer hidden units
        self.activ = activ       # attention layer hidden nonlinearity

        with tf.variable_scope(name):
            # Tensorflow can define input sizes by itself
            self.enc_att = L.Dense(self.hid_size)
            self.dec_att = L.Dense(self.hid_size)
            self.out_att = L.Dense(self.hid_size)

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
            print("D", dec.shape)
            # Compute logits
            # We will use multiple logits for each attention sentence, may be it will improve score.
            a = self.out_att(self.activ(self.enc_att(enc) + tf.expand_dims(self.dec_att(dec),1)))
            # Apply mask - if mask is 0, logits should be -inf or -1e9
            # You may need tf.where
            print("a", a.shape)
            a = tf.where(inp_mask, a, tf.fill(tf.shape(a), -1e9))
            # Compute attention probabilities (softmax)
            
            probs = tf.nn.softmax(a, axis=1)
            
            # Compute attention response using enc and probs
            attn = tf.reduce_sum(tf.multiply(probs, a), axis=1)
            
            return attn, probs
        
class BasicModel:
    def __init__(self, name, inp_voc, out_voc, emb_size=64, hid_size=128):
        """
        A simple encoder-decoder model
        """
        self.name, self.inp_voc, self.out_voc = name, inp_voc, out_voc

        with tf.variable_scope(name):
            self.emb_inp = L.Embedding(len(inp_voc), emb_size)
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size)

            self.dec_start = L.Dense(hid_size)
            self.dec0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.logits = L.Dense(len(out_voc))

            # prepare to translate_lines
            self.inp = tf.placeholder('int32', [None, None])
            self.initial_state = self.prev_state = self.encode(self.inp)
            self.prev_tokens = tf.placeholder('int32', [None])
            self.next_state, self.next_logits = self.decode(self.prev_state, self.prev_tokens)
        
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :returns: initial decoder state tensors, one or many
        """
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        inp_emb = self.emb_inp(inp)
        with tf.variable_scope('enc0'):
            _, enc_last = tf.nn.dynamic_rnn(
                              self.enc0, inp_emb,
                              sequence_length=inp_lengths,
                              dtype = inp_emb.dtype)
        dec_start = self.dec_start(enc_last)
        return [dec_start]

    def decode(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, n_tokens]
        """
        [prev_dec] = prev_state
        prev_emb = self.emb_out(prev_tokens[:,None])[:,0]
        with tf.variable_scope('dec0'):
            new_dec_out, new_dec_state = self.dec0(prev_emb, prev_dec)
        output_logits = self.logits(new_dec_out)
        return [new_dec_state], output_logits

    def translate_lines(self, inp_lines, sess, inp_voc, out_voc, max_len=100):
        """
        Translates a list of lines by greedily selecting most likely next token at each step
        :returns: a list of output lines, a sequence of model states at each step
        """
        state = sess.run(self.initial_state, {self.inp: inp_voc.to_matrix(inp_lines)})
        outputs = [[self.out_voc.bos_ix] for _ in range(len(inp_lines))]
        all_states = [state]
        finished = [False] * len(inp_lines)

        for t in range(max_len):
            state, logits = sess.run([self.next_state, self.next_logits], {**dict(zip(self.prev_state, state)),
                                           self.prev_tokens: [out_i[-1] for out_i in outputs]})
            next_tokens = np.argmax(logits, axis=-1)
            all_states.append(state)
            for i in range(len(next_tokens)):
                outputs[i].append(next_tokens[i])
                finished[i] |= next_tokens[i] == self.out_voc.eos_ix
        return out_voc.to_lines(outputs), all_states
    
class AttentiveModel(BasicModel):
    def __init__(self, name, inp_voc, out_voc,
                 emb_size=64, hid_size=128, attn_size=128):
        """ Translation model that uses attention. See instructions above. """
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hid_size=hid_size

        with tf.variable_scope(name):
            
            # define model layers
            self.emb_inp = L.Embedding(len(inp_voc), emb_size)
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.attention = AttentionLayer('Attention', hid_size, hid_size, hid_size)
            self.dec_start = L.Dense(hid_size)
            self.dec0 = tf.nn.rnn_cell.GRUCell(hid_size)
            self.logits = L.Dense(len(out_voc))
            
            # prepare to translate_lines
            self.inp = tf.placeholder('int32', [None, None])
            self.initial_state = self.prev_state = self.encode(self.inp)
            self.prev_tokens = tf.placeholder('int32', [None])
            self.next_state, self.next_logits = self.decode(self.prev_state, self.prev_tokens)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        
        # encode input sequence, create initial decoder states
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        mask = tf.expand_dims(tf.cast(infer_mask(inp, self.inp_voc.eos_ix), tf.bool), axis=2)

        inp_mask = tf.tile(mask, (1, 1, self.hid_size))
        inp_emb = self.emb_inp(inp)
        
        with tf.variable_scope('enc0'):
            enc_outputs, enc_last = tf.nn.dynamic_rnn(
                              self.enc0, inp_emb,
                              sequence_length=inp_lengths,
                              dtype = inp_emb.dtype)
            
        dec_start = self.dec_start(enc_last)
        
        # apply attention layer from initial decoder hidden state
        attn, first_attn_probas = self.attention(enc_outputs,dec_start,inp_mask)
        
        # Build first state: include
        # * initial states for decoder recurrent layers
        # * encoder sequence and encoder attn mask (for attention)
        # * make sure that last state item is attention probabilities tensor
        
        first_state = [dec_start, enc_outputs, inp_mask, first_attn_probas]
        return first_state

    def decode(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        # Unpack your state: you will get tensors in the same order that you've packed in encode
        [prev_dec, enc_outputs, inp_mask, first_attn_probas] = prev_state
        
        
        # Perform decoder step
        # * predict next attn response and probas given previous decoder state
        # * use prev tokens and next attn response to update decoder states
        # * predict logits
        prev_emb = self.emb_out(prev_tokens[:,None])[:,0]
        
        next_attn_response, next_attn_probas = self.attention(enc_outputs, prev_dec, inp_mask)
        
        with tf.variable_scope('dec0'):
            dec_input = tf.concat([prev_emb, next_attn_response], axis=-1)
            new_dec_out, new_dec_state = self.dec0(dec_input, prev_dec)
        
        output_logits = self.logits(new_dec_out)
        
        # Pack new state:
        # * replace previous decoder state with next one
        # * copy encoder sequence and mask from prev_state
        # * append new attention probas
        
        next_state = [new_dec_state, enc_outputs, inp_mask, next_attn_probas]
        return next_state, output_logits









