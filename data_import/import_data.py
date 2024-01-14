from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import os
import tqdm
from sentence_transformers import SentenceTransformer

sentence_embedding_model_path='E:\ChatSys\model\EmbeddingModel\\all-MiniLM-L6-v2'

sentence_embedding = SentenceTransformer(sentence_embedding_model_path)

def to_embeddings(items):
    if(len(items)!=2):
        items.append("")
    sentence_embeddings = sentence_embedding.encode(items[0])

    return [items[0], items[1], sentence_embeddings]

def insert_data(system_name,data_path,split_type="#####"):
    client = QdrantClient("127.0.0.1", port=6333)
    collection_name = system_name
    # 创建collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    count = 0
    for root, dirs, files in os.walk(data_path):
        for file in tqdm.tqdm(files):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for text in lines:
                    text=text.strip()
                    parts = text.split(split_type)
                    item = to_embeddings(parts)
                    print(item[:2])
                    client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=[
                            PointStruct(id=count, vector=item[2], payload={"title": item[0], "text": item[1]}),
                        ],
                    )
                    count += 1

if __name__ == '__main__':
    system_name="党建"
    insert_data(system_name,"./source_data",split_type='#')
