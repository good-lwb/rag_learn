import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer

# ========== 初始化全局组件 ==========
print("🔧 正在加载模型...")

# 1. 加载检索组件
model_path = "data/data_base/bge-small-zh-v1.5"
index_path = "data/data_base/huangdi_vectors.index"
chunks_path = "data/data_base/huangdi_chunks.txt"

embedding_model = SentenceTransformer(model_path)
index = faiss.read_index(index_path)
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

# 2. 加载LLM模型
model_llm_path = "../Qwen/Qwen2.5-7B-Instruct"
llm_model = AutoModelForCausalLM.from_pretrained(
    model_llm_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_llm_path)

print("✅ 所有模型加载完成\n")


# ========== 核心功能函数 ==========
def search(query, top_k=5):
    """检索相关文本块（自动去重）"""
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0].astype('float32')
    distances, indices = index.search(np.array([query_embedding]), top_k)

    # 去重并保留最高分结果
    unique_chunks = {}
    for idx, score in zip(indices[0], distances[0]):
        chunk = chunks[idx]
        if chunk not in unique_chunks or score > unique_chunks[chunk]:
            unique_chunks[chunk] = score

    # 按分数排序返回
    sorted_results = sorted(unique_chunks.items(), key=lambda x: -x[1])
    return "\n".join([chunk for chunk, _ in sorted_results])


def generate_response(query, chat_history=[]):
    """生成带上下文记忆的回答"""
    # 1. 检索增强
    search_results = search(query)

    # 2. 构建Prompt（包含历史对话）
    prompt_template = """
[相关背景知识]
{search_results}
************************
[对话历史]
{history}
************************
[当前问题]
{query}

请根据背景知识和对话历史回答问题：
"""

    history_str = "\n".join([f"用户：{q}\n助手：{a}" for q, a in chat_history[-3:]])  # 保留最近3轮
    prompt = prompt_template.format(
        search_results=search_results,
        history=history_str,
        query=query
    )

    # 3. LLM生成
    messages = [
        {"role": "system", "content": "你是中医助手Qwen，请专业且友好地回答用户问题。"},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([inputs], return_tensors="pt").to(llm_model.device)

    generated_ids = llm_model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7  # 控制创造性
    )
    response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)

    return response.strip()


# ========== 多轮对话循环 ==========
def chat_loop():
    print("\n🟢 中医问答系统已启动（输入'退出'结束对话）")
    history = []

    while True:
        try:
            # 1. 获取用户输入
            query = input("\n👤 user：")
            if query.lower() in ["退出", "exit", "quit"]:
                print("🛑 对话结束")
                break

            # 2. 生成回答
            print("\n🤖 正在思考...")
            response = generate_response(query, history)
            print(f"🔍 检索到的知识片段：\n{search(query)}\n")
            print(f"💡 assistant：{response}")

            # 3. 更新历史
            history.append((query, response))

        except KeyboardInterrupt:
            print("\n🛑 用户中断对话")
            break
        except Exception as e:
            print(f"❌ 发生错误：{str(e)}")
            continue


if __name__ == "__main__":
    chat_loop()