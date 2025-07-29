import os
import json
import torch
import logging
from tqdm import tqdm 
from ragas import evaluate
from datasets import Dataset
from datetime import datetime
from ragas.llms import llm_factory
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
from ragas.metrics import (faithfulness, answer_relevancy, context_recall, context_precision)


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

        return {
            "child_context": "\n\n".join(d.page_content for d in child_docs),
            "parent_context": "\n\n".join(d.page_content for d in parent_docs if d),
            "question": question
        }
        
    return retrieve_docs | prompt | llm | StrOutputParser()

def auto_evaluate(retriever, llm, embeddings):
    # 首先需要构建一个测试数据集
    questions = [
        "春季养生建议的作息时间是怎样的？",
        "夏季应该保持怎样的心情？",
        "肝气功能紊乱会导致什么表现？",
        "大肠小肠功能失常导致什么问题？",
        "眼泪产生的根本源头是什么？",
        "哭而无泪说明什么生理状态？",
        "阳刺法的具体操作方法是怎样的？",
        "狂病与癫病的针刺区别在哪里？",
        "心咳的典型表现？",
        "东方对应何脏？",
        "问诊的关键要点？"
    ]

    ground_truths = [
        "入夜即睡眠，早些起身",
        "保持愉快，切勿发怒",
        "言语增多，情绪失控",
        "清浊不分，泄泻不止",
        "源于肾精所化水液，受心神调控，志悲时失去制约而外溢",
        "心神未被真正触动，心肾不交，精气未离于目故无泪",
        "中间直刺一针配合左右斜刺四针，专治寒热病证，出针时需令微量出血",
        "狂病泻阳脉邪气至分肉皆热；癫病需持续针刺分肉经脉直至发作停止",
        "心痛喉梗，咽肿闭塞",
        "东方属木通于肝",
        "静室细问，察神观色"
    ]
    answers = []
    contexts = []
    # Inference
    chain = compose_chain(retriever, llm)
    for query in tqdm(questions, desc="Evaluating RAG performance"):
        answers.append(chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])


    # To dict
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    # ragas原生的评估是使用gpt需要openai的key，在这里指定local_llm设定成自己的模型评估,或者舍弃faithfulness,answer_relevancy这俩参数这俩都是需要api的剩下俩不需要。
    local_llm = load_model("remote") 
    # 评测结果
    result = evaluate(
        dataset = dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=local_llm,
        embeddings=embeddings
    )

    df = result.to_pandas()

    return df

if __name__ == "__main__":
    llm = load_model("remote")  #"remote" 或 "local"
    embeddings = load_embedding("local") #"remote" 或 "local"与前面split时候一致
    retriever = load_embedding_parent("../Huangdi_v4.0/data_process/chroma_db", embeddings)
    
    # 使用增强版chain
    # chain = compose_chain(retriever, llm)
    # response = chain.invoke("古人为什么长寿？")
    # print(response)
    res_evaluate = auto_evaluate(retriever, llm, embeddings)

    # 保存为 JSON（结构化，易读）
    json_path = "evaluation_result.json"
    res_evaluate.to_json(json_path, orient="records", force_ascii=False, indent=4)
    print(f"评估结果已保存到 JSON 文件: {json_path}")
