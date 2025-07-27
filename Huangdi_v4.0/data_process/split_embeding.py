import os
import json
import torch
from typing import List
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma, Milvus
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings

def load_embedding(source: str):
    """
    加载嵌入模型
    
    参数:
        source (str): 指定模型来源，'remote' 或 'local'
    
    返回:
        对应的嵌入模型实例
        
    示例:
        >>> embedding = load_embedding("local")
    """
    if source == "remote":
        # 初始化大语言模型(阿里embeding服务，在线付费token计费)
        DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
            
        # 创建嵌入模型
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=DASHSCOPE_API_KEY
        )
        return embeddings

    elif source == "local":
        # 加载本地embedding模型
        model_path = "../../bge-small-zh-v1.5"  # 修改为HuggingFace模型ID或正确本地路径
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建LangChain兼容的嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings

    else:
        raise ValueError("请指定正确的嵌入模型来源：'remote' 或 'local'")
 
def parent_splitter_database(embeddings, load_path=None):
    """
    创建主文档分割器和子文档分割器，并初始化向量数据库。
    
    参数:
        embeddings: 嵌入模型
        load_path (str): 持久化路径，None表示内存存储
    
    返回:
        tuple: 包含主文档分割器、子文档分割器、向量数据库和检索器的元组
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！"]  # 中文友好分隔符
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=20,
        separators=["\n", "。", "！", "？"]
    )

    # 创建持久化的向量数据库对象chroma
    vectorstore = Chroma(
        collection_name="split_parents", 
        embedding_function=embeddings,
        persist_directory=load_path  # None时不会持久化
    )

    # 创建内存存储对象
    store = InMemoryStore()
    
    # 创建父文档检索器
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 2}
    )

    return parent_splitter, child_splitter, vectorstore, retriever


def load_and_split_documents(file_path: str) -> List[Document]:
    """
    加载并分割章节式文本文件,这个函数是因为我们的文本是章节式的，并且在txt中是按照换行拆分的，所以需要将每个章节分割成一个Document对象。
    （具体任务具体分析）
    参数:
        file_path: 文本文件路径
    
    返回:
        分割后的Document列表（每个章节一个Document）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        chapters = [line.strip() for line in f if line.strip()]
    
    return [
        Document(
            page_content=chapter,
            metadata={
                "chapter": f"第{i+1}章",
                "parent": f"parent_{i}"  # 新增：唯一父文档标识
            }
        )
        for i, chapter in enumerate(chapters)
    ]

if __name__ == "__main__":
    # 加载嵌入模型
    embeddings = load_embedding("local")
    load_path = "./chroma_db"
    
    # 初始化分割器和数据库
    _, _, vectorstore, retriever = parent_splitter_database(embeddings, load_path)
    
    # 加载并分割文档
    docs = load_and_split_documents("../../data/huangdi_data.txt")
    
    try:
        # 添加文档集
        retriever.add_documents(docs)
        
        # 打印统计信息
        collection = vectorstore.get()
        print(f"总文档块数: {len(collection['ids'])}")
        print(f"前5个块ID示例: {collection['ids'][:5]}")
        
        # 持久化存储
        if load_path:
            vectorstore.persist()

            # 保存 docstore 内容（父文档）
            docstore_data = {
                doc.metadata["parent"]: {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            }
            with open(os.path.join(load_path, "docstore.json"), "w", encoding="utf-8") as f:
                json.dump(docstore_data, f, ensure_ascii=False, indent=2)

            print(f"父文档已保存到: {os.path.join(load_path, 'docstore.json')}")
    except Exception as e:
        print(f"处理文档时出错: {e}")
        
        
