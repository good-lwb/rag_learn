import logging
import sys
import torch
from llama_index.core import PromptTemplate, Settings, StorageContext, load_index_from_storage
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# 定义日志
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# 定义system prompt
SYSTEM_PROMPT = """You are a helpful AI assistant."""
query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

# 使用llama-index创建本地大模型
Settings.llm = HuggingFaceLLM(
    context_window = 4096,
    max_new_tokens = 2048,
    generate_kwargs = {"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt = query_wrapper_prompt,
    tokenizer_name = "../Qwen",
    model_name = "../Qwen",
    device_map = "auto",
    model_kwargs = {"torch_dtype": torch.float16},
)

# 使用LlamaDebugHandler构建事件回溯器，以追踪LlamaIndex执行过程中发生的事件
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

# 使用llama-index-embeddings-huggingface构建本地embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "../bge-base-zh-v1.5"
)

# 从存储文件中读取embedding向量和向量索引
storage_context = StorageContext.from_defaults(persist_dir="./doc_emb")

# 根据存储的embedding向量和向量索引重新构建检索索引
index = load_index_from_storage(storage_context)

# 构建查询引擎
query_engine = index.as_query_engine(similarity_top_k=5)

# 查询获得答案
response = query_engine.query("古人长寿的秘诀是什么？")
print(response)

# get_llm_inputs_outputs 返回每个LLM调用的开始/结束事件
event_pairs = llama_debug.get_llm_inputs_outputs()

# print(event_pairs[0][1].payload.keys()) # 输出事件结束时所有相关的属性

# 输出 Promt 构建过程
print(event_pairs[0][1].payload["formatted_prompt"])