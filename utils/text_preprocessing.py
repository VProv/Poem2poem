from nltk.tokenize import WordPunctTokenizer
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


def tokenize(x, tokenizer):
    return ' '.join(tokenizer.tokenize(x.lower()))

def tokenize_corpus_hw(data='data.txt', train_loc='./', bpe_voc=['en', 'ru'], num_symbols=10000):
    tokenizer = WordPunctTokenizer()
    with open(train_loc + 'train.' + bpe_voc[0], 'w') as f_src,  \
         open(train_loc + 'train.' + bpe_voc[1], 'w') as f_dst:
        for line in open(data):
            src_line, dst_line = line.strip().split('\t')
            f_src.write(tokenize(src_line, tokenizer) + '\n')
            f_dst.write(tokenize(dst_line, tokenizer) + '\n')
        
    
    # build and apply bpe vocs
    bpe = {}
    for lang in bpe_voc:
        learn_bpe(open(train_loc + 'train.' + lang), open('bpe_rules.' + lang, 'w'), num_symbols=num_symbols)
        bpe[lang] = BPE(open(train_loc + 'bpe_rules.' + lang))
        
        with open(train_loc + 'train.bpe.' + lang, 'w') as f_out:
            for line in open(train_loc + 'train.' + lang):
                f_out.write(bpe[lang].process_line(line.strip()) + '\n')
                
def create_train_bpe(train_loc, bpe_voc=['en', 'ru'], num_symbols=10000):
    """
    args:
        train_loc: location of train.lang files with previously tokenized data
    returns:
        write train.bpe.lang files into train_loc
    """
    # build and apply bpe vocs
    bpe = {}
    for lang in bpe_voc:
        print("Learning BPE...")
        learn_bpe(open(train_loc + 'train.' + lang), open(train_loc + 'bpe_rules.' + lang, 'w'), num_symbols=num_symbols)
        bpe[lang] = BPE(open(train_loc + 'bpe_rules.' + lang))
        print("Writing train files...")
        with open(train_loc + 'train.bpe.' + lang, 'w') as f_out:
            for line in open(train_loc + 'train.' + lang):
                f_out.write(bpe[lang].process_line(line.strip()) + '\n')