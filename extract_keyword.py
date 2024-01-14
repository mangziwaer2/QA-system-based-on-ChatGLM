import jieba.analyse

from stop_words import keyword_stop_words as stop_words


def extract(sentence):
    '''
    提取句子中的关键词，
    需要用到stop_words.py文件中的停用词来去除无关的关键词
    '''
    key= jieba.analyse.extract_tags(sentence, topK=10, withWeight=False, allowPOS=([]))
    key=[k for k in key if k not in stop_words]
    print("key", key)
    return key

def test_rank(sentence):
    # 提取关键词
    keywords = jieba.analyse.textrank(sentence, topK=10, withWeight=False, allowPOS=[])
    print(keywords)

def lda(sentence):
    from gensim import corpora, models

    # 准备文本数据
    documents = [sentence]

    # 分词处理
    texts = [jieba.lcut(document) for document in documents]

    # 构建词典
    dictionary = corpora.Dictionary(texts)

    # 构建文档-词频矩阵
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 训练LDA模型
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    # 提取关键词和主题
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

def rake(text):

    import nltk
    nltk.download('stopwords')
    from rake_nltk import Rake
    # 初始化RAKE
    r = Rake()

    # 提取关键词
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()[:5]

    # 打印关键词
    for keyword in keywords:
        print(keyword)

if __name__ == '__main__':
    sentence="五四运动是什么，中共二大是什么,关于诸城党建你了解什么内容"
    extract(sentence)