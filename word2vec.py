from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import word2vec, KeyedVectors

# 读取语料+断句分词(因为我的预料本来就是一句句零散的)
text_file_path = './word2vec/text_full.txt'
stop_words = stopwords.words('english')  # 得到nltk内置的所有英文停用词


f = open(text_file_path, 'r', encoding='utf-8')
line = f.readline()
sentences = []
while line:
    line = f.readline()
    words = word_tokenize(line)
    # 过滤停用词
    filter_words = [w for w in words if w not in stop_words]
    filter_words = [w for w in filter_words if len(w) > 1]
    sentences.append(filter_words)
f.close()

'''参数解释
Args:
    sentences:  是预处理后的训练语料库。是可迭代列表，但是对于较大的语料库，可以考虑直接从磁盘/网络传输句子的迭代。
    sg:         1是Skip-Gram算法，对低频词敏感；默认sg=0为CBOW算法。
    size:       是输出词向量的维数，默认值是100。这个维度的取值与我们的语料的大小相关，比如小于100M的文本语料，则使用默认值一般就可以了。
                如果是超大的语料，建议增大维度。值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，
                一般值取为100到200之间，不过见的比较多的也有300维的。
    window:     是一个句子中当前单词和预测单词之间的最大距离，window越大，
                则和某一词较远的词也会产生上下文关系。默认值为5。
                windows越大所需要枚举的预测此越多，计算的时间越长。
    min_count:  忽略所有频率低于此值的单词。默认值为5
    workers:    表示训练词向量时使用的线程数,默认是当前运行机器的处理器核数。
    还有关采样和学习率的，一般不常设置
    negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
    hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
    iter表示迭代次数，默认为5，通常设置30-50次

Returns:
    Word2Vec模型
'''
# 词向量训练
print("word2vec模型训练中...")
model = word2vec.Word2Vec(sentences, sg=0, size=256,  window=10,  min_count=20, hs=1, iter=50)

# save
# 方法1（模型不可继续训练）
model_path = './word2vec/word2vec.wv'
model.wv.save(model_path)
wv = KeyedVectors.load(model_path, mmap='r')
print(wv.word_vec('Cisco'))

# # 方法2（模型可继续训练）
# model_path = './word2vec/word2vec.model'
# model.save(model_path)
# print("Word2Vec model saved\n")
#
# model = word2vec.Word2Vec.load(model_path)
# print(model)
print("ok")
# 继续训练
# model.train([filter_words], total_examples=1, epochs=1)
