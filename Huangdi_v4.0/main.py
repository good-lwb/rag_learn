import os
import json
import torch
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.llms import Tongyi
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.vectorstores import Chroma, Milvus
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings


def load_model(source: str):
    """
    加载llm模型
    
    参数:
        source (str): 指定模型来源，'remote' 或 'local'
    
    返回:
        对应的llm模型实例
        
    示例:
        >>> llm = load_model("local")
    """
    if source == "remote":
        # 初始化大语言模型(阿里embeding服务，在线付费token计费)
        DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
            
        # 创建llm模型
        llm = Tongyi(
            model_name="qwen-max", 
            dashscope_api_key=DASHSCOPE_API_KEY
        )
        return llm

    elif source == "local":
        # 加载本地llm模型
        llm = ChatOpenAI(
            api_key="EMPTY",      # vLLM不需要key但参数必填
            base_url="http://localhost:6789/v1",  # 你的vLLM服务地址
            model="qwen",         # 必须与--served-model-name参数一致
            temperature=0.7,
            max_tokens=2048       # 控制生成长度
        )
        return llm

    else:
        raise ValueError("请指定正确的llm模型来源：'remote' 或 'local'")
 

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
        model_path = "../bge-small-zh-v1.5"  # 修改为HuggingFace模型ID或正确本地路径
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


def load_docstore_from_json(json_path: str):
    import json
    from langchain.storage import InMemoryStore

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [
        Document(page_content=v["content"], metadata=v["metadata"])
        for v in data.values()
    ]
    store = InMemoryStore()
    store.mset([(doc.metadata["parent"], doc) for doc in docs])
    return store


def load_embedding_parent(load_path, embeddings):
    vectorstore = Chroma(
        collection_name="split_parents",
        embedding_function=embeddings,
        persist_directory=load_path
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！"]
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=20,
        separators=["\n", "。", "！", "？"]
    )

    # ✅ 尝试加载 docstore.json,因为chroma不会保存父文档的分块，所以只能单独手动保存json并加载。
    docstore_path = os.path.join(load_path, "docstore.json")
    if os.path.exists(docstore_path):
        store = load_docstore_from_json(docstore_path)
        print("✅ 已加载 docstore（父文档）")
    else:
        store = InMemoryStore()
        print("⚠️ 未找到 docstore.json，docstore为空")

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3}
    )
    return retriever


def compose_chain(retriever, llm):
    template = """基于以下文档节选和完整章节回答问题, 为了增强说服力可以引用上下文片段：
    
    【相关段落】
    {child_context}
    
    【所属完整章节】
    {parent_context}
    
    问题：{question}
    答案："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    def retrieve_docs(question, save_path="retrieval_logs"):
        # 获取子文档及其父文档
        child_docs = retriever.vectorstore.similarity_search(question, k=3)
        parent_ids = [doc.metadata["parent"] for doc in child_docs]
        parent_docs = retriever.docstore.mget(parent_ids)

        # 打印子文档
        logging.info("检索到的子文档（子段落）:")
        for i, d in enumerate(child_docs):
            logging.info(f"[子文档 {i+1}]")
            logging.info(d.page_content)
            logging.info(f"所属父文档 ID: {d.metadata.get('parent', 'N/A')}")

        # 打印父文档
        logging.info("\n对应的父文档（完整章节）:")
        for i, d in enumerate(parent_docs):
            if d:
                logging.info(f"[父文档 {i+1}]")
                logging.info(d.page_content)
            else:
                logging.warning(f"[父文档 {i+1}] 内容缺失")

        # 组织输出内容
        result = {
            "question": question,
            "child_docs": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in child_docs
            ],
            "parent_docs": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata if doc else {}
                } for doc in parent_docs if doc
            ]
        }

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 用时间戳命名文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_path, f"retrieval_{timestamp}.json")

        # 保存为 JSON 文件，这部分就是方便对检索进行调试，可以通过json'来看自己检索的内容了
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logging.info(f"\n检索结果已保存至: {filename}")

        return {
            "child_context": "\n\n".join(d.page_content for d in child_docs),
            "parent_context": "\n\n".join(d.page_content for d in parent_docs if d),
            "question": question
        }
        
    return retrieve_docs | prompt | llm | StrOutputParser()

if __name__ == "__main__":
    llm = load_model("local")  #"remote" 或 "local"
    embeddings = load_embedding("local") #"remote" 或 "local"与前面split时候一致
    retriever = load_embedding_parent("./data_process/chroma_db", embeddings)
    
    # 使用增强版chain
    chain = compose_chain(retriever, llm)
    response = chain.invoke("古人为什么长寿？")
    print(response)
