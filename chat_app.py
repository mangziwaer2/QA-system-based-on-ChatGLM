import os
import chatglm_model
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from process_query import prompt


class ChatApp():
    def __init__(self,
                 chat_model_path="model/chatglm2-6b",
                 sentence_embedding_model_path='model/ChineseEmbeddingModel/all-MiniLM-L6-v2',
                 ):

        self.readable_history = []
        self.modified_history = []

        path_root = os.path.dirname(__file__)
        self.sentence_embedding = SentenceTransformer(os.path.join(path_root, sentence_embedding_model_path))

        self.client = QdrantClient("127.0.0.1", port=6333)
        self.collection_name = "党建"
        self.chat_model = chatglm_model.ChatModel(model_path=chat_model_path)

    def change_system(self,sys_name):
        self.collection_name=sys_name

    def sentence_to_embedding(self, sentence):
        if isinstance(sentence, list):
            embedding = self.sentence_embedding.encode(sentence)
        else:
            embedding = self.sentence_embedding.encode([sentence])

        return embedding[0]

    def query(self, text, max_length=2048, top_p=0.7, temperature=0.95):
        """
        执行逻辑：
        首先将输入的文本转换为向量
        然后使用Qdrant的search API进行搜索，搜索结果中包含了向量和payload
        payload中包含了title和text，title是疾病的标题，text是摘要
        最后使用chatGLM进行对话生成
        """

        sentence_embeddings = self.sentence_to_embedding(text)  # 将提问转为向量
        """
        因为提示词的长度有限，所以我只取了搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=sentence_embeddings,
            limit=3,
            search_params={"exact": False, "hnsw_ef": 128}
        )
        answers = []

        """
        因为提示词的长度有限，每个匹配的相关摘要我在这里只取了前300个字符，如果想要更多的相关摘要，可以把这里的300改为更大的值
        """
        print(text)
        for result in search_result:
            print(result.payload["title"], result.score)
            if result.score < 0.6:
                continue
            if len(result.payload["text"]) > 800:
                summary = result.payload["text"][:800]
            else:
                summary = result.payload["text"]
            answers.append({"title": result.payload["title"], "text": summary})

        if len(answers) == 0:
            answers.append({"title": "没有相关信息，请你回答:未查到相关信息", "text": ""})
        q = prompt(text, answers)
        print(q)
        output,history = self.chat_model.predict(q, max_length=max_length, top_p=top_p, temperature=temperature)

        self.modified_history.append([q, output])
        self.readable_history.append([text, output])

        if(len(self.chat_model.history)>=4):
            self.chat_model.history=self.chat_model.history[:4]

        return self.readable_history

    def stream_query(self, text, max_length=2048, top_p=0.7, temperature=0.95):
        history_len=len(self.chat_model.history)
        if (history_len>= 4):
            self.chat_model.history = self.chat_model.history[history_len-4:]
        """
        执行逻辑：
        首先将输入的文本转换为向量
        然后使用Qdrant的search API进行搜索，搜索结果中包含了向量和payload
        payload中包含了title和text，title是疾病的标题，text是摘要
        最后使用chatGLM进行对话生成
        """

        sentence_embeddings = self.sentence_to_embedding(text)  # 将提问转为向量
        """
        因为提示词的长度有限，所以我只取了搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=sentence_embeddings,
            limit=3,
            search_params={"exact": False, "hnsw_ef": 128}
        )
        answers = []

        """
        因为提示词的长度有限，每个匹配的相关摘要我在这里只取了前300个字符，如果想要更多的相关摘要，可以把这里的300改为更大的值
        """
        print(text)
        for result in search_result:
            print(result.payload["title"], result.score)
            if result.score < 0.6:
                continue
            if len(result.payload["text"]) > 800:
                summary = result.payload["text"][:800]
            else:
                summary = result.payload["text"]
            answers.append({"title": result.payload["title"], "text": summary})

        if len(answers) == 0:
            answers.append({"title": "没有相关信息，请你回答:未查到相关信息", "text": ""})
        q = prompt(text, answers)
        print(q)
        predictor = self.chat_model.predictor(q, max_length=max_length, top_p=top_p, temperature=temperature)

        modified_history=self.modified_history
        readable_history=self.readable_history

        for output, history in predictor:
            self.modified_history=modified_history+[(q,output)]
            self.readable_history=readable_history+[(text,output)]

            yield self.readable_history

    def clear_history(self):
        self.modified_history.clear()
        self.readable_history.clear()
        self.chat_model.clear_history()

    def save_history(self):
        self.chat_model.save_history()

    def load_history(self, file):
        history = self.chat_model.load_history(file)
        return history

    def get_similarity(self, s1, s2):
        e1 = self.sentence_to_embedding(s1)
        e2 = self.sentence_to_embedding(s2)

        return cos_sim(e1, e2)


if __name__ == '__main__':
    print("初始化模型....")
    model = ChatApp()
    print("模型初始化完成....")
    ans = model.query("你好")
    for a in ans:
        print(a[1])
    ans = model.query("谈谈你对五四运动的看法")
    for a in ans:
        print(a[1])
    ans = model.query("什么是中共二大")
    for a in ans:
        print(a[1])
