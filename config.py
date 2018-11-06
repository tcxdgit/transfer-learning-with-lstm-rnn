import os
from pymongo import MongoClient
import tensorflow as tf
import nlp_property
# 切换词向量请求方法，改embedding.embedding_helper   本地/服务器
from embedding.embedding_helper import EmbeddingHelper

class BaseConfig(object):
    def __init__(self):
        # Parameters
        # ==================================================
        self.batch_size = 64  # the batch_size of the training procedure
        # self.valid_num = 100  # epoch num of validation
        self.init_scale = 0.1

        # Model Hyperparameters
        self.embed_dim = 300  # embedding dim
        self.hidden_neural_size = 1000  # LSTM hidden neural size
        self.hidden_layer_num = 1  # LSTM hidden layer num
        self.num_step = 30  # max_len of training sentence

        # Misc Parameters
        self.allow_soft_placement = True  # Allow device soft device placement
        self.log_device_placement = False  # Log placement of ops on devices

        # 句子逆序输入
        self.data_reverse = True

        # 切换词向量     本地/服务器
        # self.wv = EmbeddingHelper('../work_space/vector/vector_hlt/dic_18_hlt.bin')
        self.wv = EmbeddingHelper('vector_hlt')

# config for train
class TrainConfig(BaseConfig):
    def __init__(self):
        BaseConfig.__init__(self)

        ip = nlp_property.NLP_FRAMEWORK_IP
        client = MongoClient(ip + ':27017')

        # change here !!!
        # 新华书店： bookstore
        field = 'bank_psbc'
        num_blank_intent = 9

        db = None
        if field == 'bank':
            db = client.bank
        elif field == 'bank_psbc':
            db = client.bank_psbc
        elif field == 'ecovacs':
            db = client.ecovacs
        elif field == 'suning':
            db = client.suning_biu
        else:
            pass

        if db:
            self.super_emotions = db.dialogue.distinct("super_intention")
            for i in range(num_blank_intent):
                self.super_emotions.append('')

        if field == 'suning':
            self.dataset_path = '../work_space/suning/dataset/suning_biu_combine_data'.format(field)
            self.out_dir = os.path.join(self.dataset_path, "../../module/comb_hlt/cn")
        elif field in ['ecovacs', 'bank', 'bank_psbc']:
            self.dataset_path = '../work_space/{}/dataset/{}_combine_data'.format(field, field)
            self.out_dir = os.path.join(self.dataset_path, "../../module/comb_hlt/cn")
        else:
            self.dataset_path = '../work_space/{}/dataset/{}_data'.format(field, field)
            self.out_dir = os.path.join(self.dataset_path, "../../module/rnn_transfer/cn")
        # self.dataset_name = '{}_transfer'.format(field)
        # self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", self.dataset_name, "cn"))  # output directory
        # self.out_dir = os.path.join(self.dataset_path, "../../module/comb_hlt/cn")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Model Hyper-parameters
        self.valid_portion = .02  # Percentage of the data to use for valid
        self.lr = 0.1  # the learning rate
        self.lr_decay = 0.5  # the learning rate decay
        self.keep_prob = 0.5
        self.num_epoch = 60  # num of epoch
        self.max_decay_epoch = 30  # the epoch of start decay in learning rate
        self.max_grad_norm = 5  # max_grad_norm
        self.check_point_every = 5  # save checkpoint every num epoch

        # process data
        self.data_enhance = False
        self.sort_by_len = True
