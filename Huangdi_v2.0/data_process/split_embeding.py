import os
import torch
from pathlib import Path
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_text_with_splitter(text: str, embedding_model, save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        embedding_model: 嵌入模型（需是LangChain兼容的嵌入模型）
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", "", "。"],
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")

    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embedding_model)
    print("已从文本块创建知识库...")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
    
    return knowledgeBase

if __name__ == "__main__":
    text_path = Path('../../data/huangdi_data.txt')
    with open(text_path, "r", encoding='utf-8') as f:
        text = f.read()

    # 使用HuggingFaceEmbeddings包装SentenceTransformer模型
    model_path = "../../bge-small-zh-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建LangChain兼容的嵌入模型
    model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    save_dir = "./"
    knowledgeBase = process_text_with_splitter(text, model, save_path=save_dir)
