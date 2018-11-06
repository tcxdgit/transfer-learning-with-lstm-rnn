#! /usr/bin/env python

import tensorflow as tf
import inspect
import numpy as np
import os, sys
sys.path.append("..")
import os.path
from singleton import Singleton
import codecs
import time
from sys import argv
import heapq
from rnn_text_classification.config import BaseConfig
from rnn_text_classification.rnn_model import RNN_Model
from log_helper import LogHelper


class Classify_CN(BaseConfig, metaclass=Singleton):

    lh = None

    def __init__(self, module_dir, user='user'):
        BaseConfig.__init__(self)
        print("module dir: " + module_dir)

        test_config = BaseConfig()
        test_config.batch_size = 1

        checkpoint_dir = os.path.join(module_dir, "cn", "checkpoints")
        print('checkpoint_dir:' + checkpoint_dir)
        classes_file = codecs.open(os.path.join(module_dir, "cn", "classes"), "r", "utf-8")
        self.classes = list(line.strip() for line in classes_file.readlines())
        classes_file.close()
        # print(self.classes)

        self.user = user

        # Evaluation
        # ==================================================
        # change

        # self.embedding_dim_cn = test_config.embed_dim
        # self.sentence_words_num = test_config.num_step

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/cpu:0"):
                session_conf = tf.ConfigProto(
                    allow_soft_placement=test_config.allow_soft_placement,
                    log_device_placement=test_config.log_device_placement
                    )
                session_conf.gpu_options.allow_growth = True
                self.session = tf.Session(config=session_conf)

                # with tf.name_scope("Valid"):
                with tf.variable_scope("Model"):
                    self.model = RNN_Model(config=test_config, num_classes=len(self.classes), is_training=False)

                restored_saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path.split('/')[-1])
                    restored_saver.restore(self.session, ckpt.model_checkpoint_path)
                else:
                    print('There is no saved model!!!')

        this_file = inspect.getfile(inspect.currentframe())
        dir_name = os.path.abspath(os.path.dirname(this_file))
        self.chat_log_path = os.path.join(dir_name, '..', 'log/module/rnn_classify')
        if not os.path.exists(os.path.join(self.chat_log_path, user)):
            if not os.path.exists(self.chat_log_path):
                os.makedirs(self.chat_log_path)
            f = open(self.chat_log_path+'/' + user, 'w', encoding='utf-8')
            f.close()

        if not self.lh:
            self.lh = LogHelper(user, self.chat_log_path)

    def __enter__(self):
        print('Classify_CN enter')

    def __exit__(self):
        print('Classify_CN exit')
        self.session.close()

    def generate_mask(self, set_x):
        masked_x = np.zeros([self.num_step, len(set_x)])
        for i, x in enumerate(set_x):
            x_list = x.split(' ')
            if len(x_list) < self.num_step:
                masked_x[0:len(x_list), i] = 1
            else:
                masked_x[:, i] = 1
        return masked_x

    def getCategory(self, sentence):
        # wh = WordCutHelper(0)
        # value = wh.getWords(sentence)
        # sentence = ' '.join(value)
        #
        # start_time = time.time()
        sentence_raw = sentence
        if self.data_reverse:
            sen_list = sentence.strip().split(' ')
            sen_list.reverse()
            sentence = ' '.join(sen_list)
        else:
            pass

        x_test = [sentence]

        mask_x = self.generate_mask(x_test)

        sentence_embedded_chars = self.wv.embedding_lookup(len(x_test), self.num_step, self.embed_dim, x_test)

        fetches = [self.model.prediction, self.model.probability, self.model.scores]
        feed_dict = {
            self.model.embedded_x: sentence_embedded_chars,
            self.model.mask_x: mask_x
        }

        state = self.session.run(self.model._initial_state)
        for i, (c, h) in enumerate(self.model._initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        [predictions, probabilities, scores] = self.session.run(fetches, feed_dict)

        num_classes = len(self.classes)

        if num_classes > 5:
            num_top = 5
        else:
            num_top = num_classes

        top_indexes = heapq.nlargest(num_top, range(len(scores[0])), scores[0].take)
        top_classes = []
        top_scores = []
        top_probabilities = []
        for i in range(num_top):
            top_scores.append(float(scores[0][top_indexes[i]]))
            top_classes.append(self.classes[top_indexes[i]])
            top_probabilities.append(float(probabilities[0][top_indexes[i]]))

        prediction_index = predictions[0]
        max_score = scores[0][prediction_index]
        class_prediction = self.classes[prediction_index]
        max_probability = probabilities[0][prediction_index]

        result = {'sentence': sentence_raw,
                  'score': float(max_score),
                  'probability': float(max_probability),
                  'value': class_prediction.strip(),
                  'top5_value': top_classes,
                  'top5_score': top_scores,
                  'top5_probability': top_probabilities
                  }
        return result

if __name__ == '__main__':
    script, module_path = argv
    # module_path = '../work_space/ecovacs/module/comb_hlt'
    classify = Classify_CN(module_path)
    from tools.word_cut import WordCutHelper
    wh = WordCutHelper(1)
    while 1:
        s = input('sentence: ')
        if not s:
            break

        value = wh.getWords(s)
        s = ' '.join(value)
        print(s)

        tic = time.time()
        r = classify.getCategory(s)
        toc = time.time()
        print(r)
        print(r['value'])
        print('cost time: {} s'.format(toc-tic))
        # print('Time used: {} s'.format(time_cost))
