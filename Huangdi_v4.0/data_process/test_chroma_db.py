import os
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

def test_db(db_path: str, embedding_path: str = "../../bge-small-zh-v1.5"):
    """
    向量数据库测试脚本
    
    参数:
        db_path: Chroma数据库目录路径
        embedding_path: 本地模型路径或HuggingFace模型ID
    """
    # 1. 加载嵌入模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 2. 连接现有数据库
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="split_parents"
    )
    
    print(f"\n{'='*50}")
    print(f"开始测试Chroma数据库：{db_path}")
    print(f"{'='*50}\n")
    
    # 3. 基础信息检查
    collection = vectorstore.get()
    print("[基础信息]")
    print(f"总文档块数: {len(collection['ids'])}")
    print(f"元数据字段: {list(collection['metadatas'][0].keys()) if collection['metadatas'] else '无'}")
    
    # 4. 显示样本内容
    print(f"\n[样本内容检查]")
    sample_ids = collection['ids'][:3]  # 取前3个样本
    samples = vectorstore.get(ids=sample_ids, include=["documents", "metadatas"])
    for 序号, (文档id, 内容, 元数据) in enumerate(zip(samples['ids'], samples['documents'], samples['metadatas'])):
        print(f"\n样本{序号+1} (ID: {文档id})")
        print(f"章节: {元数据.get('chapter', '无章节信息')}")
        print(f"内容预览: {内容[:100]}...")  # 显示前100字符
    
    # 5. 检索功能测试
    print(f"\n[检索功能测试]")
    测试查询列表 = ["黄帝", "岐伯", "阴阳"]  # 可修改为你的关键词
    
    for 查询词 in 测试查询列表:
        print(f"\n查询: 『{查询词}』")
        结果列表 = vectorstore.similarity_search(查询词, k=2)  # 取最相关的2个结果
        print(f"共找到 {len(结果列表)} 个相关结果")
        
        for 结果序号, 文档 in enumerate(结果列表, 1):
            print(f"\n▶ 结果{结果序号}:")
            print(f"章节: {文档.metadata.get('chapter', '未知章节')}")
            print(f"内容: {文档.page_content[:150]}...")  # 显示前150字符
    
    print("\n测试完成！")

if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    db_path = "./chroma_db"  # 你的Chroma数据库路径
    embedding_path = "../../bge-small-zh-v1.5"  # 本地路径或HF模型ID
    
    # 执行测试
    test_db(
        db_path=db_path,
        embedding_path=embedding_path
    )