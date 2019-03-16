from tqdm import tqdm_notebook
import re
import time

def parse_OpenSubtitles(path):
    """
    args:
        path: path to .tmx file
    returns:
        en_list, ru_list: lists with sentences
    """
    en_list = []
    ru_list = []
    counter = 0
    start = time.time()
    regex_corner = re.compile('[<>?]')
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if ":lang=\"en\"" in line:
                good_tokens = regex_corner.sub(' ', line).split()[3:-2]
                en_list.append(' '.join(good_tokens))
                counter += 1
            elif ":lang=\"ru\"" in line:
                good_tokens = regex_corner.sub(' ', line).split()[3:-2]
                ru_list.append(' '.join(good_tokens))
                counter += 1
            if counter % 100005 == 0:
                print(counter, time.time()-start)
                counter += 1
    return en_list, ru_list


def get_sonnets_lines_parallel(fname):
    """
    args:
        'fname': path to parallel sonnets corpus file in following format:
        one line per sonnet line
        each sonnet is represented by pair 'English-Russian'
        English and Russian version within pair are separated by line '---'
        Each version consists of multiple lines.
        Different pairs are separated by line '==='
        So, input file looks like:
        sonnet1_english_line1
        sonnet1_english_line2
        ...
        sonnet1_english_line14
        ---
        sonnet1_russian_line1
        sonnet1_russian_line2
        ...
        sonnet1_russian_line14
        ===
        sonnet2_english_line1
        sonnet2_english_line2
        ...
        sonnet2_english_line14
        ---
        sonnet2_russian_line1
        sonnet2_russian_line2
        ...
        sonnet2_russian_line14
        ===
        ...

    returns:
        'sonnet_lines_english': all lines of all english sonnets in a single list
        'sonnet_lines_russian': corresponding lines of russian sonnets
        'non_fourteen_lines': proper sonnet should contain 14 lines. Otherwise we skip it. 
        We count such cases in 'non_fourteen_lines'

    """
    sonnet_lines_english = []
    sonnet_lines_russian = []
    non_fourteen_lines = 0
    with codecs.open(fname, mode = 'r', encoding = 'utf-8') as f:
        sonnets = f.read().replace('\r', '').split('\n===\n')
    for sonnet in sonnets:
        en, ru = sonnet.split('\n---\n')
        en = [line.strip() for line in en.split('\n') if line.strip() != '']
        ru = [line.strip() for line in ru.split('\n') if line.strip() != '']
        if len(en) != 14 or len(ru) != 14:
            non_fourteen_lines += 1
            continue
        sonnet_lines_english.extend(en)
        sonnet_lines_russian.extend(ru)
    assert len(sonnet_lines_english) == len(sonnet_lines_russian)
    return sonnet_lines_english, sonnet_lines_russian, non_fourteen_lines
