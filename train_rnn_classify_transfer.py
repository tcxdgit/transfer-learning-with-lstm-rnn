# coding:utf-8

import sys
sys.path.append("..")
import tensorflow as tf
import os
import time
from rnn_text_classification.rnn_model import RNN_Model
# from rnn_text_classification import data_helper_comb as data_helper
from rnn_text_classification import data_helper
import codecs
import re
from rnn_text_classification.config import TrainConfig as Config

class TrainRNNTransfer(Config):
    super_emotions = None

    def __init__(self):
        Config.__init__(self)
        if self.super_emotions:
            print("Super intention: {}".format(self.super_emotions))

    def evaluate(self, config, model,session,data):
        correct_num = 0
        total_num = len(data[0])
        for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(data,batch_size=config.batch_size)):

            x_embedded = self.wv.embedding_lookup(len(list(x)), config.num_step, config.embed_dim, list(x), 1)

            fetches = model.correct_num
            feed_dict={}
            feed_dict[model.embedded_x] = x_embedded
            feed_dict[model.target] = y
            feed_dict[model.mask_x] = mask_x

            state = session.run(model._initial_state)
            for i, (c, h) in enumerate(model._initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            count = session.run(fetches, feed_dict)
            correct_num += count

        accuracy = float(correct_num)/total_num
        return accuracy

    def train_step(self):

        # Load data
        print("Loading data...")

        config = Config()

        valid_config = Config()
        valid_config.keep_prob = 1.0
        valid_config.batch_size = 32

        classify_files = []
        classify_names = []
        for parent, dirnames, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                # 这里是为了避免存在缓存文件
                if filename[-1] == '~':
                    # os.remove(os.path.join(dataset_path, filename))
                    continue
                else:
                    classify_files.append(os.path.join(self.dataset_path, filename))
                    classify_names.append(filename)

        # common, 不加上级意图
        # load_data(classify_files, config, sort_by_len=True, enhance = True, reverse=True)
        train_data, valid_data = data_helper.load_data(classify_files, config, sort_by_len=True,
                                                       enhance=self.data_enhance, reverse=self.data_reverse)

        # classify_names = ['Expect', 'Love', 'Joy', 'Surprise', 'Hate', 'Anger', 'Sorrow', 'Anxiety']
        # train_data, valid_data = data_helper.load_data(dataset_path, classify_names, config,
        #                                                enhance=data_enhance, reverse=data_reverse)

        num_classes = len(classify_names)

        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        print(checkpoint_dir)

        # print("begin training")
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=config.allow_soft_placement,
                log_device_placement=config.log_device_placement)
            session = tf.Session(config=session_conf)

            initializer = tf.random_uniform_initializer(-1 * config.init_scale, 1 * config.init_scale)

            with tf.name_scope("Train"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    model = RNN_Model(config=config, num_classes=num_classes, is_training=True)

            with tf.name_scope("Valid"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    valid_model = RNN_Model(config=valid_config, num_classes=num_classes, is_training=False)

                # 记录验证集精度变化情况
                acc_valid = tf.placeholder(tf.float32)
                dev_summary = tf.summary.scalar('dev_accuracy', acc_valid)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.check_point_every)
            session.run(tf.global_variables_initializer())

            if os.path.exists(checkpoint_dir):
                classes_file = codecs.open(os.path.join(config.out_dir, "classes"), "r", "utf-8")
                classes = list(line.strip() for line in classes_file.readlines())
                classes_file.close()

                # 类别是否发生改变
                if sorted(classify_names) == sorted(classes):
                    print('-----continue training-----')

                    new_classify_files = []
                    for c in classes:
                        idx = classify_names.index(c)
                        new_classify_files.append(classify_files[idx])

                    # classify_files = new_classify_files

                    restored_saver = tf.train.Saver(tf.global_variables())
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        print('restore model: '.format(ckpt.model_checkpoint_path))
                        restored_saver.restore(session, ckpt.model_checkpoint_path)
                    else:
                        print('-----train from beginning-----')
                else:
                    print('-----change network-----')
                    not_restore = ['softmax_w:0', 'softmax_b:0']
                    restore_var = [v for v in tf.global_variables() if v.name.split('/')[-1] not in not_restore]
                    restored_saver = tf.train.Saver(restore_var)
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        print('restore model: '.format(ckpt.model_checkpoint_path))
                        restored_saver.restore(session, ckpt.model_checkpoint_path)
                    else:
                        pass

                    classes_file = codecs.open(os.path.join(config.out_dir, "classes"), "w", "utf-8")
                    for classify_name in classify_names:
                        classes_file.write(classify_name)
                        classes_file.write('\n')
                    classes_file.close()
            else:
                print('-----train from begin-----')
                os.makedirs(checkpoint_dir)
                classes_file = codecs.open(os.path.join(config.out_dir, "classes"), "w", "utf-8")
                for classify_name in classify_names:
                    classes_file.write(classify_name)
                    classes_file.write('\n')
                classes_file.close()

            train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

            valid_summary_dir = os.path.join(valid_config.out_dir, "summaries", "valid")
            valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, session.graph)

            # add checkpoint
            # checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            global_steps = 1
            # 保存的第一个模型是global_steps为1的，验证集的accuracy大于acc_max时开始替换
            acc_max = 0.0
            begin_time = int(time.time())

            try:
                for i in range(config.num_epoch):
                    print("the %d epoch training..." % (i+1))
                    lr_decay = config.lr_decay ** max(i-config.max_decay_epoch, 0)
                    model.assign_new_lr(session, config.lr*lr_decay)

                    # 加上级意图
                    # train_data, valid_data = data_helper.load_data(
                    #     classify_files, self.super_emotions, num_classes, config,
                    #     sort_by_len=True, enhance=self.data_enhance, reverse=self.data_reverse)

                    # run epoch
                    for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(train_data, batch_size=config.batch_size)):
                        feed_dict = {}
                        # word embedding
                        x_embedded = self.wv.embedding_lookup(len(list(x)), config.num_step, config.embed_dim, list(x),
                                                              0)
                        feed_dict[model.embedded_x] = x_embedded
                        feed_dict[model.target] = y
                        feed_dict[model.mask_x] = mask_x
                        fetches = [model.cost, model.accuracy, model.train_op, model.summary]
                        state = session.run(model._initial_state)
                        for i, (c, h) in enumerate(model._initial_state):
                            feed_dict[c] = state[i].c
                            feed_dict[h] = state[i].h
                        cost, accuracy, _, summary = session.run(fetches, feed_dict)
                        if train_summary_writer:
                            train_summary_writer.add_summary(summary, global_steps)
                            train_summary_writer.flush()
                        print("the %i step, train cost is: %f and the train accuracy is %f" %
                              (global_steps, cost, accuracy))

                        # accuracy on valid setting
                        if global_steps % 100 == 0:
                            valid_accuracy = self.evaluate(valid_config, valid_model, session, valid_data)
                            print("the {} step, train cost is: {}, the train accuracy is {} and the valid accuracy is {}\n".
                                format(global_steps, cost, accuracy, valid_accuracy))

                            val_summary = session.run(dev_summary, {acc_valid: valid_accuracy})
                            if valid_summary_writer:
                                valid_summary_writer.add_summary(val_summary, global_steps)
                                valid_summary_writer.flush()

                            # i=0: 第一个epoch
                            if valid_accuracy > acc_max:
                                acc_max = valid_accuracy
                                # if (i % config.check_point_every == 0) or (i + 1 == config.num_epoch):

                            for f_name in os.listdir(checkpoint_dir):
                                if 'model' in f_name:
                                    os.remove(os.path.join(checkpoint_dir, f_name))
                            path = saver.save(session, checkpoint_prefix, global_steps)
                            print("Saved model checkpoint to{}\n".format(path))

                        global_steps += 1
                        del x_embedded
            except KeyboardInterrupt:
                pass

            # 修改checkpoint文件中的model路径
            lines = []
            with open(os.path.join(checkpoint_dir, "checkpoint"), "r") as f:
                f_lines = list(f.readlines())
                for i in range(len(f_lines)):
                    if i in [0, len(f_lines)-1]:
                        line_deal = self.replace_model_path(f_lines[i])
                        lines.append(line_deal)

            with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
                for line in lines:
                    f.write(line)

            # 删除旧的summary文件
            self.remove_old_file(train_summary_dir)
            self.remove_old_file(valid_summary_dir)

            print("the train is finished")
            end_time = int(time.time())
            print("training takes {} seconds already\n".format(end_time-begin_time))
            print("program end!")

    def replace_model_path(self, checkpoint_path):
        pat = re.compile(r'\"(/.+/)model-\d+\"$')
        model = ''.join(pat.findall(checkpoint_path))
        text = re.sub(model, '', checkpoint_path)
        return text

    def remove_old_file(self, dir_name):
        lists = os.listdir(dir_name)  # 列出目录的下所有文件和文件夹保存到lists
        # print(lists)
        lists.sort(key=lambda fn: os.path.getmtime(dir_name + "/" + fn))  # 按时间排序
        file_new = os.path.join(dir_name, lists[-1])  # 获取最新的文件保存到file_new
        # print(file_new)

        paths = [os.path.join(dir_name, file_name) for file_name in lists]

        for p in paths:
            if p == file_new:
                pass
            else:
                os.remove(p)

if __name__ == "__main__":
    Train = TrainRNNTransfer()
    Train.train_step()
    print('done')
