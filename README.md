# 介绍
    使用qdrant和chatGLM搭建的党建问答助手
## 前期准备

本服务需要使用Qdrant向量数据库，所以需要先安装Qdrant，为了方便可以使用docker启动：
`docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
`
启动数据库后运行data_import文件夹中的数据添加程序

## 模型文件
chat_app.py中可以指明模型路径

1、ChatGLM模型 https://aistudio.baidu.com/datasetdetail/246362 将模型文件下载并解压到model文件夹下

2、句子向量化模型 使用sentenceTransformer 放在model/EmbeddingModel文件夹下


## 关于Qdrant向量数据库

你可以查看Qdrant的官方文档：https://qdrant.tech/documentation/


## 运行
`
python test_with_web.py
`
