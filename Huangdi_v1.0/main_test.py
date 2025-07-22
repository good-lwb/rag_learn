import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer

def search(query, top_k=10):
    # 编码查询语句（显示进度）
    # print(f"\n🔍 正在搜索: '{query}'")
    query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype('float32')

    distances, indices = index.search(np.array([query_embedding]), top_k)

    results_list =  [(chunks[idx], float(score)) for idx, score in zip(indices[0], distances[0])]

    chunk_sum = set()
    for i, (chunk, score) in enumerate(results_list):
        chunk_sum.add(chunk)

    chunk_str = "\n".join(list(chunk_sum))
    return chunk_str

# results = search("古代人为什么长寿？")
# print(results)

def llm(model_path, query):
    # print("🔧 正在加载LLM模型...")
    # model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    search_results = search(query, top_k=10)
    prompt = f"""[{search_results}]
    **************************
    请根据以上[]中的的内容并结合你自身的知识，回答下面的问题：
    {query}
    """

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


if __name__ == "__main__":

    with open("data/data_base/huangdi_chunks.txt", "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f if line.strip()]

    model_path = "data/data_base/bge-small-zh-v1.5"
    index_path = "data/data_base/huangdi_vectors.index"
    model_llm_path = "../Qwen/Qwen2.5-7B-Instruct"

    model = SentenceTransformer(model_path)
    index = faiss.read_index(index_path)
    query = "古代人为什么长寿？"
    response = llm(model_llm_path, query)
    print(response)