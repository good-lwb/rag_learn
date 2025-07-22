import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import re
import faiss
from tqdm import tqdm

# 加载中文Embedding模型
print("🔧 正在加载Embedding模型...")
model_path = "D:/Project/Blm/RAG/ChineseMedicine_QA/data/data_base/bge-small-zh-v1.5"
model = SentenceTransformer(model_path)
print("✅ 模型加载完成\n")

def split_paragraphs(text):
    return [p.strip() for p in text.split('\n') if p.strip()]
def split_sentences(paragraph):
    """分句：中文标点分割（简单版）"""
    return [s.strip() for s in re.split(r'[。！？；]', paragraph) if s.strip()]

def semantic_chunk(text, max_chunk_size=500, min_similarity=0.75):
    print("📖 文本分句中...")
    # 1. 初步分段和分句
    paragraphs = split_paragraphs(text)
    all_sentences = []
    for para in paragraphs:
        all_sentences.extend(split_sentences(para))

    # 2. 生成句子嵌入
    print("\n🔢 生成句子向量...")
    embeddings = model.encode(all_sentences)

    # 3. 动态聚类分块（基于相似度）
    print("\n🧩 语义分块合并中...")
    chunks = []
    current_chunk = []
    current_embedding = None

    for sentence, embedding in zip(all_sentences, embeddings):
        if not current_chunk:
            current_chunk.append(sentence)
            current_embedding = embedding
        else:# 计算与当前块的相似度
            sim = np.dot(embedding, current_embedding)/(np.linalg.norm(embedding) * np.linalg.norm(current_embedding))

            if sim >= min_similarity and len("".join(current_chunk)) + len(sentence) <= max_chunk_size:
                current_chunk.append(sentence)
                current_embedding = (current_embedding * (len(current_chunk) - 1) + embedding)
                current_embedding /= len(current_chunk)
            else:
                chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_embedding = embedding

    if current_chunk:
        chunks.append("".join(current_chunk))
    return chunks

print("📂 读取文本文件中...")
with open("../data_process/huangdi_data.txt", "r", encoding="utf-8") as f:
    texts = f.read()
    text = texts.replace("\n\n","\n")
print(f"✅ 已读取文本（长度：{len(text)} 字符）\n")
    # print(text)

chunks = semantic_chunk(text)
# for i, chunk in enumerate(chunks):
#     print(f"【Chunk {i + 1}】\n{chunk}\n{'-' * 50}")

print("\n🏗️ 构建向量数据库中...")
# 生成所有块的向量
embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True, batch_size=32)  # 归一化向量
dimension = embeddings.shape[1]

# 构建Faiss数据库
print("  正在构建FAISS索引...")
index = faiss.IndexFlatIP(dimension)  # 使用内积（余弦相似度）
index.add(embeddings.astype('float32'))  # FAISS需要float32格式

#保存分词结果
chunks_file = "huangdi_chunks.txt"
with open(chunks_file, "w", encoding="utf-8") as f:
    f.write("\n".join(chunks))
print(f"✅ 分块文本已保存至 {chunks_file}")

# 3. 保存数据库到磁盘
faiss.write_index(index, "huangdi_vectors.index")
print(f"✅ 向量数据库已保存（包含 {len(chunks)} 条数据）\n")

