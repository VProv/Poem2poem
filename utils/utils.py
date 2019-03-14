import sys
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

class Vocab:
    def __init__(self, tokens, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        """
        A special class that converts lines of tokens into matrices and backwards
        """
        assert all(tok in tokens for tok in (bos, eos, unk))
        self.tokens = tokens
        self.token_to_ix = {t:i for i, t in enumerate(tokens)}
        self.bos, self.eos, self.unk = bos, eos, unk
        self.bos_ix = self.token_to_ix[bos]
        self.eos_ix = self.token_to_ix[eos]
        self.unk_ix = self.token_to_ix[unk]

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_lines(lines, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        flat_lines = '\n'.join(list(lines)).split()
        tokens = sorted(set(flat_lines))
        tokens = [t for t in tokens if t not in (bos, eos, unk) and len(t)]
        tokens = [bos, eos, unk] + tokens
        return Vocab(tokens, bos, eos, unk)

    def tokenize(self, string):
        """converts string to a list of tokens"""
        tokens = [tok if tok in self.token_to_ix else self.unk
                  for tok in string.split()]
        return [self.bos] + tokens + [self.eos]

    def to_matrix(self, lines, max_len=None):
        """
        convert variable length token sequences into  fixed size matrix
        example usage:
        >>>print( as_matrix(words[:3],source_to_ix))
        [[15 22 21 28 27 13 -1 -1 -1 -1 -1]
         [30 21 15 15 21 14 28 27 13 -1 -1]
         [25 37 31 34 21 20 37 21 28 19 13]]
        """
        lines = list(map(self.tokenize, lines))
        max_len = max_len or max(map(len, lines))

        matrix = np.zeros((len(lines), max_len), dtype='int32') + self.eos_ix
        for i, seq in enumerate(lines):
            row_ix = list(map(self.token_to_ix.get, seq))[:max_len]
            matrix[i, :len(row_ix)] = row_ix

        return matrix

    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        :param crop: if True, crops BOS and EOS from line
        :return:
        """
        lines = []
        for line_ix in map(list,matrix):
            if crop:
                if line_ix[0] == self.bos_ix:
                    line_ix = line_ix[1:]
                if self.eos_ix in line_ix:
                    line_ix = line_ix[:line_ix.index(self.eos_ix)]
            line = ' '.join(self.tokens[i] for i in line_ix)
            lines.append(line)
        return lines


### Utility TF functions ###


def infer_length(seq, eos_ix, time_major=False, dtype=tf.int32):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: lengths, int32 vector of shape [batch]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos_ix), dtype)
    count_eos = tf.cumsum(is_eos,axis=axis,exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos,0),dtype),axis=axis)
    return lengths


def infer_mask(seq, eos_ix, time_major=False, dtype=tf.float32):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    axis = 0 if time_major else 1
    lengths = infer_length(seq, eos_ix, time_major=time_major)
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(seq)[axis], dtype=dtype)
    if time_major: mask = tf.transpose(mask)
    return mask


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.
    :param values: logits for all actions: float32[batch,tick,action]
    :param indices: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]
    batch_i = tf.tile(tf.range(0,batch_size)[:, None],[1,seq_len])
    time_i = tf.tile(tf.range(0,seq_len)[None, :],[batch_size,1])
    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values,indices_nd)


def save(variables, path, sess=None):
    """
    saves variable weights independently (without tf graph)
    :param variables: an iterable of TF variables
    """
    sess = sess or tf.get_default_session()
    assert sess is not None, "please make sure you defined a default TF session"
    var_values = sess.run({w.name : w for w in variables})
    np.savez(path, **var_values)


def load(variables, path, sess=None, verbose=True):
    """
    loads variable weights saved with save function above
    :param variables: a list/tuple of 
    """
    sess = sess or tf.get_default_session()
    assert sess is not None, "please make sure you defined a default TF session"
    var_values = np.load(path)
    var_values = {name: var_values[name] for name in var_values}
    not_initialized = []
    ops = []
    for var in variables:
        if var.name in var_values:
            ops.append(tf.assign(var, var_values.pop(var.name)))
        else:
            not_initialized.append(var.name)
    sess.run(ops)
    if verbose:
        if len(var_values):
            print('Checkpoint weights not used:', ' '.join(var_values.keys()), file=sys.stderr)
        if len(not_initialized):
            print('Variables not initialized:', ' '.join(not_initialized), file=sys.stderr)
    


def initialize_uninitialized(sess=None):
    """
    Initialize unitialized variables, doesn't affect those already initialized
    :param sess: in which session to initialize stuff. Defaults to tf.get_default_session()
    """
    sess = sess or tf.get_default_session()
    assert sess is not None, "please make sure you defined a default TF session"
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        
def compute_logits(model, inp, out, **flags):
    """
    :param inp: input tokens matrix, int32[batch, time]
    :param out: reference tokens matrix, int32[batch, time]
    :returns: logits of shape [batch, time, voc_size]
    
    * logits must be a linear output of your neural network.
    * logits [:, 0, :] should always predic BOS
    * logits [:, -1, :] should be probabilities of last token in out
    This function should NOT return logits predicted when taking out[:, -1] as y_prev
    """
    batch_size = tf.shape(inp)[0]
    
    # Encode inp, get initial state
    first_state = model.encode(inp, **flags)
    
    # initial logits: always predict BOS
    first_logits = tf.log(tf.one_hot(tf.fill([batch_size], model.out_voc.bos_ix),
                                     len(model.out_voc)) + 1e-30)
    
    
    # Decode step
    def step(blob, y_prev):
        # Given previous state, obtain next state and next token logits
        # prev_tokens int vector of batch size??
        # return state and logits
        prev_state, logits = blob
        state, logits = model.decode(prev_state, y_prev)
        return [state, logits]
      
    # Decode step
    #def step(prev_state, y_prev):
        # Given previous state, obtain next state and next token logits
        # prev_tokens int vector of batch size??
        # return state and logits

    #    return model.decode(prev_state, y_prev)

    # You can now use tf.scan to run step several times.
    # use tf.transpose(out) as elems (to process one time-step at a time)
    # docs: https://www.tensorflow.org/api_docs/python/tf/scan
    
    
    _, logits_seq = tf.scan(step, 
           elems=tf.transpose(out)[:-1],
           initializer=[first_state, first_logits])
    
    # prepend first_logits to logits_seq
    logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)
    
    # Make sure you convert logits_seq from [time, batch, voc_size] to [batch, time, voc_size]
    logits_seq = tf.transpose(logits_seq, [1, 0, 2])
    
    return logits_seq

def compute_loss(model, inp, out, out_voc, **flags):
    """
    Compute loss (float32 scalar) as in the formula above
    :param inp: input tokens matrix, int32[batch, time]
    :param out: reference tokens matrix, int32[batch, time]
    
    In order to pass the tests, your function should
    * include loss at first EOS but not the subsequent ones
    * divide sum of losses by a sum of input lengths (use infer_length or infer_mask)
    """
    mask = infer_mask(out, out_voc.eos_ix)    
    logits_seq = compute_logits(model, inp, out, **flags)
    
    # Compute loss as per instructions above
    print(mask.shape)
    print(logits_seq.shape)
    sparce_softmax = tf.losses.sparse_softmax_cross_entropy(labels=out, logits=logits_seq)
    return tf.reduce_sum(sparce_softmax * mask) / tf.reduce_sum(mask)

def compute_bleu(model, inp_lines, out_lines, sess, inp_voc, out_voc, bpe_sep='@@ ', **flags):
    """ Estimates corpora-level BLEU score of model's translations given inp and reference out """
    translations, _ = model.translate_lines(inp_lines, sess, inp_voc, out_voc, **flags)
    # Note: if you experience out-of-memory error, split input lines into batches and translate separately
    return corpus_bleu([[ref] for ref in out_lines], translations) * 100
