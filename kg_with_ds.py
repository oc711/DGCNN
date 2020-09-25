#! -*- coding:utf-8 -*-


from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
import json
import time
from nltk import word_tokenize
import numpy as np
from random import choice
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
import ahocorasick
import codecs
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, GRU, Bidirectional
from keras.layers import Add, Lambda, Embedding, Dropout, Concatenate
from keras.optimizers import Adam
from keras.layers import Layer
from keras.callbacks import Callback, EarlyStopping


# 函数
def tokenize(s):
    return [i for i in word_tokenize(s)]

def position_id(x):
    '''
        生成pid：（batch_size, maxlen）
        shape和x一样，（batch_size, 0 ~ maxlen-1）
    '''
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])  # (MLB,)
    pid = K.expand_dims(pid, 0)  # (1,MLB)
    pid = K.tile(pid, [K.shape(x)[0], 1])  # (batch_size, MLB)
    ret = K.abs(pid - K.cast(r, 'int32'))  # TODO： 为什么预测op的时候要把s的首尾位置减去
    return ret


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """
        膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]  # char_size
    h = Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate)(seq)    #(batch_size, ?, char_size * 2)

    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h

    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq



def seq_maxpool(x):
    """
        seq是[None, seq_len, s_size]的格式，
        mask是[None, seq_len, 1]的格式，先除去mask部分，
        然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1, keepdims=True)


def get_k_inter(x, n=6):
    seq, k1, k2 = x     # seq = t = shape(None(batch_size), None（seq_len）, 128); k1=k2= shape(batch_size, 1)
    k_inter = [K.round(k1 * a + k2 * (1 - a)) for a in np.arange(n) / (n - 1.)]  # list [六个shape为（batch_size, 1）的Tensor]
    k_inter = [seq_gather([seq, k]) for k in k_inter]   # #list, 6[batch_size, ?. char_size(128)]
    k_inter = [K.expand_dims(k, 1) for k in k_inter]     # list, 6[batch_size, 1, ?, char_size(128)]
    k_inter = K.concatenate(k_inter, 1)  # [batch_size, 6, ?, char_size(128)]
    return k_inter


def seq_gather(x):
    """
        seq是[None, seq_len, s_size]的格式，
        idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
        最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    ret = K.tf.gather_nd(seq, idxs)
    return ret



def random_generate(d, spo_list_key):
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]    # 随机选择一个spo
        k = np.random.randint(len(predicates[spi[1]]))  # 查询predicates中的选择的spo，spo数量，随机选择一个
        spo = predicates[spi[1]][k]
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]   # 将原数据中的so，使用predicates中so替换掉
        return {'text': text, spo_list_key: spo_list}



def sent2vec(S):
    """
        S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])



def extract_items(text_in):
    text_words = tokenize(text_in.lower())
    text_in = ''.join(text_words)
    pre_items = {}
    for sp in spoer.extract_items(text_in):
        subjectid = text_in.find(sp[0])     # 找到主语起idx
        objectid = text_in.find(sp[2])      # 找到宾语起idx
        if subjectid != -1 and objectid != -1:
            key = (subjectid, subjectid + len(sp[0]))
            if key not in pre_items:
                pre_items[key] = []
            pre_items[key].append((objectid,
                                   objectid + len(sp[2]),
                                   predicate2id[sp[1]]))
    _pres = np.zeros((len(text_in), 2))
    for j in pre_items:
        _pres[j[0], 0] = 1
        _pres[j[1] - 1, 1] = 1
    _pres = np.expand_dims(_pres, 0)
    R = []
    _t1 = [char2id.get(c, 1) for c in text_in]
    _t1 = np.array([_t1])
    _t2 = sent2vec([text_words])
    _k1, _k2 = subject_model.predict([_t1, _t2, _pres])
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.4)[0], np.where(_k2 > 0.3)[0]
    _subjects, _PREO = [], []

    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j + 1]
            _subjects.append((_subject, i, j))
            _preo = np.zeros((len(text_in), num_classes, 2))
            for _ in pre_items.get((i, j + 1), []):
                _preo[_[0], _[2], 0] = 1
                _preo[_[1] - 1, _[2], 1] = 1
            _preo = _preo.reshape((len(text_in), -1))
            _PREO.append(_preo)

    if _subjects:
        """
            如果在当前句子抽取到了合适的主语subjects，则进行后续的抽取objects和predicate，
            如果连主语都找不到，则该句子就直接返回空[]。
        """
        _PRES = np.repeat(_pres, len(_subjects), 0)
        _PREO = np.array(_PREO)
        _t1 = np.repeat(_t1, len(_subjects), 0)     # 刚才得到的_t1=[1, sent_len]，在第0维重复，则_t1=[len(_subjects), sent_len]
        _t2 = np.repeat(_t2, len(_subjects), 0)     # t2同理。意思是每个主语都要提取它的object和predicate
        _k1, _k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2, -1, 1))     # 变换后，_k1和_k2的shape=[len(_subjects), 1]
        _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2, _PRES, _PREO])     # 通过 object_model 输出宾语的起始、终止位置概率_k1, _k2
        for i, _subject in enumerate(_subjects):
            _oo1, _oo2 = np.where(_o1[i] > 0.3), np.where(_o2[i] > 0.2)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1: _ooo2 + 1]
                        _predicate = id2predicate[_c1]
                        R.append((_subject[0], _predicate, _object))
                        break

        spo_list = set()
        for s, p, o in R:
            spo_list.add((s, p, o))
        return list(spo_list)
    else:
        return []

# 类
class SpoSearcher:
    def __init__(self, train_data):
        self.s_ac = AC_Unicode()
        self.o_ac = AC_Unicode()
        self.sp2o = {}
        self.spo_total = {}
        for i, d in tqdm(enumerate(train_data), desc=u'构建三元组搜索器'):
            for s, p, o in d['spo_list']:
                self.s_ac.add_word(s, s)
                self.o_ac.add_word(o, o)
                if (s, o) not in self.sp2o:
                    self.sp2o[(s, o)] = set()
                if (s, p, o) not in self.spo_total:
                    self.spo_total[(s, p, o)] = set()
                self.sp2o[(s, o)].add(p)
                self.spo_total[(s, p, o)].add(i)
        self.s_ac.make_automaton()
        self.o_ac.make_automaton()

    def extract_items(self, text_in, text_idx=None):
        R = set()
        for s in self.s_ac.iter(text_in):
            for o in self.o_ac.iter(text_in):
                if (s[1], o[1]) in self.sp2o:
                    for p in self.sp2o[(s[1], o[1])]:
                        if text_idx is None:
                            R.add((s[1], p, o[1]))
                        elif self.spo_total[(s[1], p, o[1])] - set([text_idx]):
                            R.add((s[1], p, o[1]))
        return list(R)

class AC_Unicode:
    """稍微封装一下，弄个支持unicode的AC自动机
    """

    def __init__(self):
        self.ac = ahocorasick.Automaton()

    def add_word(self, k, v):
        # k = k.encode('utf-8')    # 报错 TypeError: expected string
        return self.ac.add_word(k, v)

    def make_automaton(self):
        return self.ac.make_automaton()

    def iter(self, s):
        # s = s.encode('utf-8')  # 报错 TypeError: expected string
        return self.ac.iter(s)



class Attention(Layer):
    """
        多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head  # 8
        self.size_per_head = size_per_head  # 16
        self.out_dim = nb_head * size_per_head  # 128
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # print(input_shape) # [TensorShape([None, None, 128]), TensorShape([None, None, 128]), TensorShape([None, None, 128]), TensorShape([None, None, 1]), TensorShape([None, None, 1])]
        super(Attention, self).build(input_shape)
        q_in_dim = input_shape[0][-1]   # 128
        k_in_dim = input_shape[1][-1]   # 128
        v_in_dim = input_shape[2][-1]   # 128
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]    # (batch_size, ?, char_size)
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel)  # shape=(None, None, num_heads*size_per_head), dtype=float32)
        kw = K.dot(k, self.k_kernel)  # shape=(None, None, 128), dtype=float32)
        vw = K.dot(v, self.v_kernel)  # shape=(None, None, 128), dtype=float32)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))  # shape=(None, None, 8, 16), dtype=float32)
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))  # shape=(None, None, 8, 16), dtype=float32)
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))  # shape=(None, None, 8, 16), dtype=float32)
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))  # shape=(None, 8, None, 16), dtype=float32)
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))  # shape=(None, 8, None, 16), dtype=float32)
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))  # shape=(None, 8, None, 16), dtype=float32)
        # Attention
        '''参数：
            x：ndim >= 2的Keras张量或变量。
            y：ndim >= 2的Keras张量或变量。
            axes：具有目标维度的（或单个）int列表。axes[0]和axes[1]的长度应该是相同的。axis[0]表示x参数参与dot的维；axis[1]表示y参数参与dot的维
        '''
        a = K.batch_dot(qw, kw, axes=[3, 3])  # tf2.2.0+keras 2.4.0结果的shape=(None, 8, None, 8, None), dtype=float32), 莫名其妙多出来的最后一维, 正常需要 a shape = (batch_size, 8, ?, ?)，各种debug不行以后只能降低版本
        # a = K.reshape(a, (-1, self.nb_head, K.shape(a)[1], K.shape(a)[1]))  # 变化后 a的shape=(None, 8, None, None), dtype=float32)
        a = a / self.size_per_head ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))  # 变化后 a的shape=(None, None, None, 8), dtype=float32)
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))  # 变换后 a shape ：(batch_size, 8, ?, ?)
        a = K.softmax(a)  # 变换后 a shape ：(batch_size, 8, ?, ?)

        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])   # shape=(None, 8, None, 8, 16), dtype=float32), keras 版本的问题导致计算结果多了一维度，正常需要的输出是shape=(None, 8, None, 16)
        # o = K.reshape(o, (-1, self.nb_head, K.shape(a)[1], self.size_per_head))  # 变化后 o的shape=(None, 8, None, 16), dtype=float32)
        o = K.permute_dimensions(o, (0, 2, 1, 3))  # 变化后 o的shape=(batch_size, ?, 8, 16)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))  # 变化后 o的shape=(batch_size, ?, num_heads*size_per_head), dtype=float32)
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)




class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """

    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))

    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))

    def reset_old_weights(self):
        """重置模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))



class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        if len(self.data) % self.batch_size == 0:
            self.steps = len(self.data) // self.batch_size
        else:
            self.steps = len(self.data) // self.batch_size + 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))  # 否则下一行报错
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list'  # if np.random.random() > 0.5 else 'spo_list_with_pred'
                d = random_generate(self.data[i], spo_list_key)     # 随机将原数据中的so替换为predicates中相同p的so
                text = d['text'][:maxlen]    # 限制最大长度
                text_words = tokenize(text)     # nltk分词
                text = ''.join(text_words)
                items = {}
                for sp in d[spo_list_key]:
                    subjectid = text.find(sp[0])    # 返回s的位置
                    objectid = text.find(sp[2])     # 返回o的位置
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid,
                                           objectid + len(sp[2]),
                                           predicate2id[sp[1]]))
                pre_items = {}
                for sp in spoer.extract_items(text, i):
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        if key not in pre_items:
                            pre_items[key] = []
                        pre_items[key].append((objectid,
                                               objectid + len(sp[2]),
                                               predicate2id[sp[1]]))
                if items:
                    T1.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                    T2.append(text_words)
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1
                    pres = np.zeros((len(text), 2))
                    for j in pre_items:
                        pres[j[0], 0] = 1
                        pres[j[1] - 1, 1] = 1
                    # k1, k2 = np.array(items.keys()).T     # 报错 TypeError: iteration over a 0-d array
                    k1, k2 = np.array(list(items.keys())).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])       # TODO：这里的k2和k1可能不是一组吧
                    o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))
                    for j in items.get((k1, k2), []):
                        o1[j[0], j[2]] = 1
                        o2[j[1] - 1, j[2]] = 1
                    preo = np.zeros((len(text), num_classes, 2))
                    for j in pre_items.get((k1, k2), []):
                        preo[j[0], j[2], 0] = 1
                        preo[j[1] - 1, j[2], 1] = 1
                    preo = preo.reshape((len(text), -1))
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2 - 1])
                    O1.append(o1)
                    O2.append(o2)
                    PRES.append(pres)
                    PREO.append(preo)
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)     # text，字序列
                        T2 = sent2vec(T2)   # text，词序列
                        S1 = seq_padding(S1)     # s，s起始位置标注序列
                        S2 = seq_padding(S2)     # s，s结束位置标注序列
                        O1 = seq_padding(O1, np.zeros(num_classes))     # op，o起始位置标注+pid
                        O2 = seq_padding(O2, np.zeros(num_classes))     # op，o结束位置标注+pid
                        K1, K2 = np.array(K1), np.array(K2)     # s，s的起始位置和结束位置
                        PRES = seq_padding(PRES, np.zeros(2))   # 远程监督的结果
                        PREO = seq_padding(PREO, np.zeros(num_classes * 2))  # 远程监督的结果
                        yield [T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO], None
                        T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []



class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """
            第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('./save/best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        '''模型使用Adam优化器进行训练，先用10−3的学习率训练不超过50个epoch，
            然后加载训练的最优结果，再用10−4的学习率继续训练到最优。
            第一个epoch用来WarmUp，如果不进行WarmUp可能不收敛。
            为了保证训练结果稳定提升，模型用到了EMA（Exponential Moving Average），衰减率为0.9999。
        '''
        EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
                self.stage == 0 and epoch > 10 and
                (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            train_model.load_weights('./save/best_model.weights')
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, learning_rate/10)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))

    def evaluate(self):
        orders = ['subject', 'predicate', 'object']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('output/test_pred.json', 'w')
        for d in tqdm(iter(test_data)):
            R = set(extract_items(d['text']))
            # T = set(d['spo_list'])
            T = set()  # 真实的ground truth spo_list
            for item in d['spo_list']:  # 将列表中的list元素逐一读取即可
                T.add((item[0], item[1], item[2]))

            A += len(R & T)  # 交集 = True Positive
            B += len(R)  # 预测出的spo数量
            C += len(T)  # 真实的ground truth spo数量

            s = json.dumps({
                'text': d['text'],
                'spo_list': [
                    dict(zip(orders, spo)) for spo in T
                ],
                'spo_list_pred': [
                    dict(zip(orders, spo)) for spo in R
                ],
                'new': [
                    dict(zip(orders, spo)) for spo in R - T
                ],
                'lack': [
                    dict(zip(orders, spo)) for spo in T - R
                ]
            }, ensure_ascii=False, indent=4)
            # F.write(s.encode('utf-8') + '\n')
            F.write(str(s.encode('utf-8')) + '\n')  # 这是因为encode返回的是bytes型的数据, write函数参数需要为str类型，转化为str即可

        F.close()
        return 2 * A / (B + C), A / B, A / C


def test(test_data):
    """输出测试结果
    """
    # orders = ['subject', 'predicate', 'object', 'object_type', 'subject_type']
    orders = ['subject', 'predicate', 'object']

    F = open('output/test_pred.json', 'w')

    for d in tqdm(iter(test_data)):
        R = set(extract_items(d['text']))
        s = json.dumps({
            'text': d['text'],
            'spo_list': [
                dict(zip(orders, spo + ('', ''))) for spo in R
            ]
        }, ensure_ascii=False)
        # F.write(s.encode('utf-8') + '\n')
        F.write(str(s.encode('utf-8')) + '\n')  # 这是因为encode返回的是bytes型的数据, write函数参数需要为str类型，转化为str即可
    F.close()


# 参数
mode = 0
char_size = 256
maxlen = 512
dropout_rate = 0.25
batch_size = 64
learning_rate = 1e-3
min_learning_rate = 1e-5

# 数据
# word2vec_model_path = './word2vec/word2vec.model'
word2vec_model_path = './word2vec/word2vec.wv'
total_data_path = './data/data_train_me.json'
schemas_path = './data/all_schemas_me.json'
all_chars_path = './data/all_chars_me.json'

# 词向量
# word2vec = Word2Vec.load(word2vec_model_path)
word2vec = KeyedVectors.load(word2vec_model_path, mmap='r')

id2word = {i + 1: j for i, j in enumerate(word2vec.index2word)}
word2id = {j: i for i, j in id2word.items()}

word2vec = word2vec.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])

# 加载数据
total_data = json.load(open(total_data_path))
id2predicate, predicate2id = json.load(open(schemas_path))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open(all_chars_path))  # 字母切割
num_classes = len(id2predicate)

# 训练数据预打乱
if not os.path.exists('./random_order_vote.json'):
    # random_order = range(len(total_data)) # 报错 TypeError: ‘range’ object does not support item assignment
    random_order = list(range(len(total_data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('./random_order_vote.json', 'w'), indent=4)
else:
    random_order = json.load(open('./random_order_vote.json'))

# 训练集：验证集=8：2
train_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 != mode]
test_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 == mode]

# 将训练集中所有的 predicate 汇总到一起
predicates = {}  # 格式：{predicate: [(subject, predicate, object)]}
for d in train_data:
    for spo in d['spo_list']:
        if spo[1] not in predicates:
            predicates[spo[1]] = []
        predicates[spo[1]].append(spo)

spoer = SpoSearcher(train_data)

# 模型
# 输入
t1_in = Input(shape=(None,))
t2_in = Input(shape=(None, word_size))
s1_in = Input(shape=(None,))
s2_in = Input(shape=(None,))
k1_in = Input(shape=(1,))
k2_in = Input(shape=(1,))
o1_in = Input(shape=(None, num_classes))
o2_in = Input(shape=(None, num_classes))
pres_in = Input(shape=(None, 2))
preo_in = Input(shape=(None, num_classes * 2))

t1, t2, s1, s2, k1, k2, o1, o2, pres, preo = t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in, pres_in, preo_in
mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t1)   # 通过比较x(在idx=2的位置增加一维)、0两个值的大小来输出对错, mask:(batch_size, seq_length, 1)


# 位置编码
# 输入t1=[samples（None）, sequence_length（None）]的2D张量
pid = Lambda(position_id)(t1)  # 返回pid:(batch_size, MLB)
position_embedding = Embedding(maxlen, char_size, embeddings_initializer='zeros')
pv = position_embedding(pid)  # (batch_size, MLB, char_size)


# token_ids
# 输入t1=[samples（None）, sequence_length（None）]的2D张量
t1 = Embedding(len(char2id) + 2, char_size)(t1)  # 0: padding, 1: unk, 输出t1=(batch_size, ?, char_size)
t2 = Dense(char_size, use_bias=False)(t2)  # 词向量也转为同样维度, t2 = (batch_size, ?, char_size)
t = Add()([t1, t2, pv])  # 字向量、词向量、位置向量相加,(batch_size, ?, char_size)
t = Dropout(dropout_rate)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask])     # (batch_size, ?, char_size)
# 12层的DGCNN
t = dilated_gated_conv1d(t, mask, 1)  # (batch_size, ?, char_size)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)
t_dim = K.int_shape(t)[-1]  # 128=char_size
'''
    包括代码中的变量pn1、pn2、pc、po，这些模块从理论上提供了一些“全局信息”，
    pn1、pn2可以认为是全局的实体识别模块，
    而pc可以认为是全局的关系检测模块，
    po可以认为是全局的关系存在性判断，
    这些模块都不单独训练，而是直接乘到s、o的预测结果上。
'''
pn1 = Dense(char_size, activation='relu')(t)
pn1 = Dense(1, activation='sigmoid')(pn1)       # (batch_size, ?, 1)
pn2 = Dense(char_size, activation='relu')(t)
pn2 = Dense(1, activation='sigmoid')(pn2)       # (batch_size, ?, 1)
################################### Attention 部分 ##########################################
# 自注意力机制
# mask =(batch_size, seq_length, 1)的形状
h = Attention(8, 16)([t, t, t, mask])  # (batch_size, ?, num_heads*size_per_head)
h = Concatenate()([t, h, pres])  # (batch_size, ?, char_size+num_heads*size_per_head+2)
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)  # (batch_size, ?, char_size)
ps1 = Dense(1, activation='sigmoid')(h)   # (batch_size, ?, 1)
ps2 = Dense(1, activation='sigmoid')(h)   # (batch_size, ?, 1)
ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pn1])     # (batch_size, ?, 1)
ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pn2])     # (batch_size, ?, 1)

subject_model = Model([t1_in, t2_in, pres_in], [ps1, ps2])  # 预测subject的模型

t_max = Lambda(seq_maxpool)([t, mask])  # (batch_size，1,char_size)
pc = Dense(char_size, activation='relu')(t_max)  # (batch_size，1,char_size)
pc = Dense(num_classes, activation='sigmoid')(pc)   # (batch_size，1,5) cause num_classes=5
##################################### BiLSTM 部分 ###########################################
"""
    t_dim = 128
    t = shape(None, None, 128)
    k1=k2= shape(None, 1)
"""
k = Lambda(get_k_inter, output_shape=(6, t_dim))([t, k1, k2])   # k =  (batch_size,6,char_size) 才是正确的
k = Bidirectional(GRU(t_dim))(k)    # [batch_size, char_size*2]
##################################### 位置编码 部分 ###########################################

k1v = position_embedding(Lambda(position_id)([t, k1]))  # [batch_size, ？, char_size]
k2v = position_embedding(Lambda(position_id)([t, k2]))  # [batch_size, ？, char_size]
kv = Concatenate()([k1v, k2v])       # [batch_size, ？, char_size*2]

k = Lambda(lambda x: K.expand_dims(x[0], 1) + x[1])([k, kv])    # [batch_size, ？, char_size*2]

h = Attention(8, 16)([t, t, t, mask])
h = Concatenate()([t, h, k, pres, preo])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)   # (batch_size, ?, char_size)
po = Dense(1, activation='sigmoid')(h)
po1 = Dense(num_classes, activation='sigmoid')(h)
po2 = Dense(num_classes, activation='sigmoid')(h)
po1 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po1, pc, pn1])
po2 = Lambda(lambda x: x[0] * x[1] * x[2] * x[3])([po, po2, pc, pn2])

object_model = Model([t1_in, t2_in, k1_in, k2_in, pres_in, preo_in], [po1, po2])  # 输入text和subject，预测object及其关系
train_model = Model([t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in, pres_in, preo_in], [ps1, ps2, po1, po2])

s1 = K.expand_dims(s1, 2)   # (batch_size, seq_length, 1)
s2 = K.expand_dims(s2, 2)   # (batch_size, seq_length, 1)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

o1_loss = K.sum(K.binary_crossentropy(o1, po1), 2, keepdims=True)
o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
o2_loss = K.sum(K.binary_crossentropy(o2, po2), 2, keepdims=True)
o2_loss = K.sum(o2_loss * mask) / K.sum(mask)

loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()

EMAer = ExponentialMovingAverage(train_model)
EMAer.inject()

train_D = data_generator(train_data, batch_size=batch_size)
evaluator = Evaluate()

if __name__ == '__main__':
    starttime = time.perf_counter()
    # es = EarlyStopping(monitor='val_loss', patience=4, verbose=2, mode='min')
    history = train_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=120,
        verbose=1,
        callbacks=[evaluator]
    )
    endtime = time.perf_counter()
    print("Training time = {} s\n".format(endtime - starttime))
    # savehist('./save/history.json', history)
    test(test_data)
else:
    train_model.load_weights('./save/best_model.weights')
