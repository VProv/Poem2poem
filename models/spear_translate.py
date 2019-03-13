import tensorflow as tf
import keras.layers as L
import numpy as np
from models.utils import infer_length, infer_mask
from models.utils import select_values_over_last_axis, compute_logits, compute_loss, compute_bleu
from models.attention_model import BasicModel, AttentionLayer


class SpearTranslate(BasicModel):
    def __init__(self, name, inp_voc, out_voc, config,
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
            with tf.variable_scope("forward"):
                self.enc0 = tf.nn.rnn_cell.LSTMCell(hid_size, forget_bias=1.0, state_is_tuple=False)
            with tf.variable_scope("backward"):
                self.enc1 = tf.nn.rnn_cell.LSTMCell(hid_size, forget_bias=1.0, state_is_tuple=False)
                
            self.attention = AttentionLayer('Attention', hid_size *4, hid_size, hid_size)
            self.dec_start = L.Dense(hid_size*2)
            
            self.dec0 = tf.nn.rnn_cell.LSTMCell(hid_size, forget_bias=1.0, state_is_tuple=False)
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
        #inp_mask = tf.cast(infer_mask(inp, self.inp_voc.eos_ix), tf.bool)
        inp_emb = self.emb_inp(inp)
        
        with tf.variable_scope('enc0'):
            enc_outputs_t, enc_last_t = tf.nn.bidirectional_dynamic_rnn(
                              self.enc0, self.enc1, inp_emb,
                              sequence_length=inp_lengths,
                              dtype=inp_emb.dtype)
            #[batch, time, hid_size*4]
            enc_outputs = tf.concat(enc_outputs_t, 2)
            #[batch, hid_size * 4]
            enc_last = tf.concat(enc_last_t, 1)
            
        dec_start = self.dec_start(enc_last)
        
        # apply attention layer from initial decoder hidden state
        attn, first_attn_probas = self.attention(enc_outputs, dec_start, inp_mask)
        
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
