import sys
sys.path.append("..")
from sys import argv
import json
from IMQ import IMessageQueue
from rnn_text_classification.rnnclassify_transfer import Classify_CN

class ClassifyRNNMQ(Classify_CN):

    mq = None #messagequeue

    def __init__(self, model_path, embed_way, publish_key, receive_key):
        Classify_CN.__init__(self, model_path, embed_way)
        self.mq = IMessageQueue(receive_key, publish_key, receive_key, receive_key,  self.getCategory_mq)

    def getCategory_mq(self, key, sentence, publish_func=''):
        r = Classify_CN.getCategory(self, sentence)
        if publish_func and r:
            publish_func(json.dumps(r), key.replace('request', 'reply'))
        else:
            print('[ClassifyRNN] no publish')

if __name__ == '__main__':
    script, model_name, embed_way, model_path = argv
    bc = ClassifyRNNMQ(model_path, embed_way, '', 'nlp.classify.rnn.' + model_name + '.request.#')
