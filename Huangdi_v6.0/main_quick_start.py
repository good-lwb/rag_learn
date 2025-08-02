import sys
import torch
import logging
from transformers import BitsAndBytesConfig
from llama_index.core.schema import MetadataMode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle

"""
快速使用和构建llama_index的rag
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# llama_index提供的提示词模板
SYSTEM_PROMPT = """You are a helpful AI assistant."""
query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

# 设置量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,  # 启用嵌套量化，在第一轮量化之后会进行第二轮量化，为每个参数额外节省 0.4 比特
    bnb_4bit_compute_dtype = torch.bfloat16, # 更改量化模型的计算数据类型来加速训练
)

# llama_index的Setting可以修改配置使用自己的llm
Settings.llm = HuggingFaceLLM(
    context_window = 4096,
    max_new_tokens = 2048,
    generate_kwargs = {"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt = query_wrapper_prompt,
    tokenizer_name = "../Qwen",
    model_name = "../Qwen",
    device_map = "auto", #"auto","balanced","balanced_low_0","sequential"
    model_kwargs = {
        "trust_remote_code":True,
        "quantization_config": quantization_config
    }
)

# llama_index调用本地embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="../bge-small-zh-v1.5"
)

# llama_index提供的分档解析器，支持多种数据格式（pdf效果不好）
# llama_index提供了很多的文档解析器，比如llama_index.readers.web import SimpleWebPageReader能够直接根据url解析网页数据到文档（爬虫的感觉）
documents = SimpleDirectoryReader("../data", required_exts=[".txt"]).load_data()

"""
切分文档，将chunk转换成embedding向量
llama_index提供多种文本切分器
    SentenceSplitter：在切分指定长度的 chunk 同时尽量保证句子边界不被切断；
    CodeSplitter：根据 AST（编译器的抽象句法树）切分代码，保证代码功能片段完整；
    SemanticSplitterNodeParser：根据语义相关性对将文本切分为片段。
llama_index提供的向量检索库VectorStoreIndex（这点比langchain好很多，langchain就是自己构建，llama_index在向量数据库构建这方面是要优于langchain的，集成的东西很多）
这里构建的向量库只是保存在内存中没有持久化
"""
index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])

# 检索
query_engine = index.as_query_engine(streaming=True, similarity_top_k=5)

# 答案生成（单论对话）
response = query_engine.query("如何能够调节身体阴阳平衡？")
response.print_response_stream() # 流式输出打印
print()

# 多轮对话
# chat_engine = index.as_chat_engine()
# streaming_response = chat_engine.stream_chat("如何能够调节身体阴阳平衡？")
# for token in streaming_response.response_gen:
#     print(token, end="", flush=True)