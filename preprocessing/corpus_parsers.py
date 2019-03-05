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