# -*- coding: utf-8 -*-
# @Date    : 2021/1/14
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : sts-b.py
"""
use snli-fint-tuning weights as initial bert-weights, training on sts-b dataset
"""
import os
import json
from tqdm import tqdm
from scipy.stats import spearmanr

from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.models import *
from toolkit4nlp.optimizers import *

from toolkit4nlp.tokenizers import *
from toolkit4nlp.utils import *


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[5], l[6], float(l[4])))
    return D


train = load_data('/home/mingming.xu/datasets/NLP/GLUE/STS-B/sts-train.csv')
test = load_data('/home/mingming.xu/datasets/NLP/GLUE/STS-B/sts-test.csv')
dev = load_data('/home/mingming.xu/datasets/NLP/GLUE/STS-B/sts-dev.csv')

config_path = '/home/mingming.xu/pretrain/NLP/google_uncased_english_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'sbert.ckpt'  # also can use original bert/roberta/...
vocab_path = '/home/mingming.xu/pretrain/NLP/google_uncased_english_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(vocab_path, do_lower_case=True)

maxlen = 64
batch_size = 16
epochs = 5
lr = 1e-5


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        token_ids_1, segment_ids_1, token_ids_2, segment_ids_2, labels = [], [], [], [], []
        for is_end, item in self.get_sample(shuffle):
            sen1, sen2, label = item

            tokens_1, segments_1 = tokenizer.encode(sen1, maxlen=maxlen)
            tokens_2, segments_2 = tokenizer.encode(sen2, maxlen=maxlen)

            token_ids_1.append(tokens_1)
            segment_ids_1.append(segments_1)
            token_ids_2.append(tokens_2)
            segment_ids_2.append(segments_2)
            labels.append([label])

            if is_end or len(token_ids_1) == self.batch_size:
                token_ids_1 = pad_sequences(token_ids_1, maxlen=maxlen)
                segment_ids_1 = pad_sequences(segment_ids_1, maxlen=maxlen)
                token_ids_2 = pad_sequences(token_ids_2, maxlen=maxlen)
                segment_ids_2 = pad_sequences(segment_ids_2, maxlen=maxlen)
                labels = pad_sequences(labels)

                yield [token_ids_1, segment_ids_1, token_ids_2, segment_ids_2], labels
                token_ids_1, segment_ids_1, token_ids_2, segment_ids_2, labels = [], [], [], [], []


train_generator = data_generator(train, batch_size)
valid_generator = data_generator(dev, batch_size)
test_generator = data_generator(test, batch_size)


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


class CosineSimilarity(Layer):
    def call(self, inputs):
        x1, x2 = inputs
        x1 = K.l2_normalize(x1, axis=1)
        x2 = K.l2_normalize(x2, axis=1)
        x = K.sum(x1 * x2, axis=1, keepdims=True)
        return x

    def compute_output_shape(self, input_shape):
        return (None, 1)


bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, name='bert')

token_inputs_1 = Input(shape=(None,), name='x1')
segment_inputs_1 = Input(shape=(None,), name='s1')
token_inputs_2 = Input(shape=(None,), name='x2')
segment_inputs_2 = Input(shape=(None,), name='s2')

output_1 = bert([token_inputs_1, segment_inputs_1])
output_2 = bert([token_inputs_2, segment_inputs_2])

u = GlobalAveragePooling1D(name='pool_1')(inputs=output_1)
v = GlobalAveragePooling1D(name='pool_2')(inputs=output_2)

x = CosineSimilarity()([u, v])

model = Model([token_inputs_1, segment_inputs_1, token_inputs_2, segment_inputs_2], x)

infer_model = Model([token_inputs_1, segment_inputs_1], u, name='encoder')
model.summary()


def get_sentence_vector(sentences):
    token_ids, segment_ids = [], []
    for sent in sentences:
        tokens, segments = tokenizer.encode(sent, maxlen=maxlen)
        token_ids.append(tokens)
        segment_ids.append(segments)

    token_ids = pad_sequences(token_ids)
    segment_ids = pad_sequences(segment_ids)

    vec = infer_model.predict([token_ids, segment_ids], verbose=True)
    return vec


def cal_sim(data):
    # cal cosine sim
    sentences_1 = [s[0] for s in data]
    sentences_2 = [s[1] for s in data]
    vecs_1 = get_sentence_vector(sentences_1)
    vecs_2 = get_sentence_vector(sentences_2)
    vecs_1 = vecs_1 / (vecs_1 ** 2).sum(axis=1, keepdims=True) ** 0.5
    vecs_2 = vecs_2 / (vecs_2 ** 2).sum(axis=1, keepdims=True) ** 0.5
    sims = (vecs_1 * vecs_2).sum(axis=1)
    return sims


def evaluate(data):
    # 计算相关系数
    sims = cal_sim(data)
    labels = [d[-1] for d in data]
    cor = np.corrcoef(sims, labels)[0, 1]  # Pearson correlation
    spear, _ = spearmanr(sims, labels)  # Spearman rank correlation
    return cor, spear


class Evaluator(keras.callbacks.Callback):
    def __init__(self, score_type='pearson', save_path='best_model.weights'):
        self.score_type = score_type
        self.save_path = save_path
        self.best_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        p, s = evaluate(dev)
        if self.score_type == 'pearson':
            score = p
        else:
            score = s
        if score > self.best_score:
            self.best_score = score
            bert.save_weights(self.save_path)
        print('steps is: {}, score is:{}, best score is: {}'.format(epoch + 1, score, self.best_score))


Opt = extend_with_weight_decay(Adam)
exclude_from_weight_decay = ['Norm', 'bias']
Opt = extend_with_piecewise_linear_lr(Opt)
para = {
    'learning_rate': 2e-5,
    'weight_decay_rate': 0.01,
    'exclude_from_weight_decay': exclude_from_weight_decay,
    'lr_schedule': {int(len(train_generator) * 0.1 * epochs): 1, int(len(train_generator) * epochs): 0},
}

opt = Opt(**para)

model.compile(loss='MSE', optimizer=opt, metrics=['mean_squared_error'])

if __name__ == '__main__':
    save_path = 'best_model.weights'
    evaluator = Evaluator(save_path=save_path)
    model.fit(train_generator.generator(),
              epochs=epochs,
              steps_per_epoch=len(train_generator),
              callbacks=[evaluator]
              )

    bert.load_weights(save_path)
    print(evaluate(train))
    print(evaluate(test))
