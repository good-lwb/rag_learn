# 黄帝内经rag问答系统
![效果图1](https://github.com/good-lwb/rag_learn/blob/main/assets/0223ff4aaffd3d88cfc3141b5b168c67.png)
![效果图2](https://github.com/good-lwb/rag_learn/blob/main/assets/ce1dac7e50b97c166bc6e2960535b0ab.png)
本项目使用爬虫获取了皇帝内经全文以此为数据构建检索增强系统  
在本次项目中将rag分为三步：  
1. indexing索引阶段  
2. retrive检索阶段  
3. generation生成阶段


本项目主要对这三个阶段分别展开优化

~~下面是不同版本的项目说明~~  
## 项目说明  
**v1.0**  
1.0版本旨在快速启动rag体验rag流程并且在代码中使用了大量打印注释方便理解rag流程  
*索引*：使用最简单的按500token分块，未作overlap，bge-samll-zh-v1.5作为embeding模型，使用faiss构架本地向量数据库  
*检索*：使用最基础的huggingface加载器加载模型推理，未做多路召回，未做rerank，无增强检索技术

**v2.0**  
2.0版本旨在使用基本的优化技术（非企业级）  
提供了api版本（main_api.py）和本地加载模型的版本（main.py）  
*索引*：使用langchain.text_spliter对文档进行分块chunk=512、overlap=128，embeding='bge-samll-zh-v1.5'，向量数据库=faiss  
*检索*：最简单的双路召回（关键词检索(BM25)、向量匹配），召回10+10，rerank取前5  

**v3.0**  
3.0版本使用Qwen-Rag框架进行开发（可以作为企业的快速启动测试，能在保证质量的前提下体验Rag带来的效果提升）  
这里还是提供两个版本的测试脚本，api版本（main_api.py）和本地加载模型的版本（main.py）  
对于本地加载模型，在此版本中我们直接使用vllm来加载本地模型给Qwen-Rag使用（直接在终端使用bash start_vllm.sh即可加载模型）；  
同时在这次测试中考虑到v2版本无记忆对话不支持多轮对话，也在Qwen-rag框架下进行了多轮对话的加入  


**v4.0**  
4.0版本旨在引入rag的一些测试评估效果（开发ing）  

## 如何使用
首先你需要下载bge-small-zh-v1.5;bge-reranker-base;Qwen3-4B三个模型,并保存在项目路径下.  
langchain版本>0.3即可,其余不做版本要求.   
!pip install langchain  
!pip install langchain-openai  
!pip install langchain-community  
!pip install faiss-cpu
!pip install openai
!pip install vllm
!pip install -U "qwen-agent[rag,code_interpreter,gui,mcp] #v3用的

