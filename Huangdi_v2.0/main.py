import os
import torch
import numpy as np
from langchain.llms.base import LLM
from langchain.schema import Document
from typing import Optional, List, Dict, Any
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever 
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pydantic import Field

from typing import Optional, List, Any
from pydantic import Field, PrivateAttr
from langchain.llms.base import LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenLocalLLM(LLM):
    model_path: str

    # 用 PrivateAttr 定义不参与 Pydantic 校验的私有属性
    _device: str = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, model_path: str):
        super().__init__(model_path=model_path)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "qwen_local"



def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    return knowledgeBase

def hybrid_retrieval(query: str, vector_store: FAISS, top_k_vector: int = 10, top_k_keyword: int = 10) -> List[Document]:
    """
    混合检索（向量+关键词）
    
    参数:
        query: 查询文本
        vector_store: FAISS向量数据库
        top_k_vector: 向量检索返回数量
        top_k_keyword: 关键词检索返回数量
    
    返回:
        合并后的文档列表（未排序）
    """
    # 获取FAISS中的原始文本块
    all_texts = [doc.page_content for doc in vector_store.docstore._dict.values()]

    # 1. 向量检索
    vector_docs = vector_store.similarity_search(query, k=top_k_vector)
    
    
    # 2. 关键词检索（BM25）
    bm25_retriever = BM25Retriever.from_texts(
        texts=all_texts,
        metadatas=[{"source": f"text_{i}"} for i in range(len(all_texts))]
    )
    keyword_docs = bm25_retriever.invoke(query, top_k=top_k_keyword)
    
    # 合并并去重
    seen = set()
    unique_docs = []
    for doc in vector_docs + keyword_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs

def rerank_documents(query: str, documents: List[Document], model_name: str = "BAAI/bge-reranker-base", top_n: int = 5) -> List[Document]:
    """
    使用交叉编码器对文档重排序
    
    参数:
        query: 查询文本
        documents: 待排序文档列表
        model_name: 排序模型名称
        top_n: 返回前N个文档
    
    返回:
        排序后的文档子集
    """
    # 初始化排序模型（建议缓存模型实例）
    cross_encoder = CrossEncoder(model_name)
    
    # 准备排序数据
    pairs = [[query, doc.page_content] for doc in documents]
    
    # 计算相关性分数
    scores = cross_encoder.predict(pairs)
    
    # 按分数排序
    ranked_indices = np.argsort(scores)[::-1]  # 降序排列
    
    # 返回前top_n个文档
    return [documents[i] for i in ranked_indices[:top_n]]

def generation_res(question: str, docs, chatLLM: QwenLocalLLM):  
    """
    根据召回的文档生成回复
    
    参数:
        query: 输入问题
        docs: 查询到的文档片段
    
    返回:
        rag_res: 返回生成答案
    """
    # 自定义适合中医的Prompt模板
    TCM_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        你是一名资深中医，请根据《黄帝内经》的内容专业回答下列问题。
        
        上下文：
        {context}
        
        问题：{question}
        
        要求：
        1. 不要使用文言文回复，避免用户看不懂
        2. 必须引用上下文中的理论依据
        3. 解释需符合中医基本原理
        
        答案：
        """
    )

    # 加载QA链
    chain = load_qa_chain(
        llm=chatLLM, 
        chain_type="stuff",
        prompt=TCM_PROMPT,
        verbose=True  # 显示详细执行过程
    )

    result = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    return result

if __name__ == "__main__":
    model_path = "../bge-small-zh-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model = HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    rerank_model = "../bge-reranker-base"
    # 加载向量数据库
    knowledgeBase = load_knowledge_base("./data_process", emb_model)
    # 初始化封装后的LLM
    chatLLM = QwenLocalLLM(model_path="../Qwen/")
    # 设置查询问题
    # query = "古人为什么可以长寿？"
    query = "什么导致人们常常咳嗽？"
    # 执行相似度搜索，找到与查询相关的文档
    hybrid_docs = hybrid_retrieval(query, knowledgeBase)
    # 对召回的文档进行重排序
    reranked_docs = rerank_documents(query, hybrid_docs, model_name=rerank_model)
    # 生成回复
    rag_res = generation_res(query, reranked_docs, chatLLM)
    print(f"AI回复:{rag_res['output_text'].strip()}")