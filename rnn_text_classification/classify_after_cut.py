import time
import sys
sys.path.append("..")
from rnn_text_classification.classify import Classify_CN
from sys import argv
from tools.word_cut import WordCutHelper

script, field, cut_way = argv

wh = None
if cut_way == 'jieba':
    wh = WordCutHelper(0)
elif cut_way == 'hlt':
    wh = WordCutHelper(1)

cc = Classify_CN(field)
while 1:
    s = input('input: ')

    value = wh.getWords(s)
    sentence = ' '.join(value)
    print(sentence)
    tic = time.time()
    result = cc.getCategory(sentence)
    toc = time.time()
    print('time:{}s'.format(float(toc - tic)))
    print(result)
