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
4.0版本中引入了父文档检索器，同时合并api加载模型和本地vllm加载的代码，并添加logging打印功能  

在这里介绍以下父文档检索器，父文档检索器主要解决，某些任务当chunk设置的比较小的时候检索到的chunk和question相似度较高，但是由于chunk较小可能导致内容信息不全，就会使模型输出不准确。而大的chunk虽然会保存较丰富的信息，但是chunk较长可能与question匹配相似度不够高。  
为了解决上述问题所以引入父文档检索器，先对文本块先分割成parent文本块（chunk较长），在分割较短的child块（chunk较短）；使用child进行相似度匹配，然后塞给模型对应的parent块，这样就能使得模型在保证相似度较高的前提下拿到更多的上下文信息。  
langchain的父文档检索器主要提供了两种方式：检索完整文档 和 检索较大的文档块，我个人更推荐第二种，这里不做科普感兴趣的朋友可以自己去看一下父文档检索器的原理和详细的介绍，我这里主要是说明一下方便使用。  

*索引*：使用langchain.retrievers提供的父文档检索器（ParentDocumentRetriever）构建Chroma本地向量数据库  
  在这部分在由于ParentDocumentRetriever不支持持久化保存父文档信息我们单独构建json将父文档chunk保存，再通过json在rag检索时进行加载，使得检索精度得到极大的提升。  
  同时，支持两中加载embedding模型的方式（api/本地），通过参数控制方便使用  
  除此之外提供test_chroma_db.py文件支持对本地chroma向量数据库进行测试  

*检索*：父文档检索器（检索top_k=3个相似度最高的子文档，并根据重构的json返回对应的父文档片段）  
  4.0版本中将llm的本地化加载合并为同一main.py中并通过指定remote/local参数进行配置使用阿里云api加载或本地部署的vllm  
*日志*：在4.0中引入了logging功能，并将检索信息通过json文件进行保存，方便后续对rag检索能进行评估  
在这部分我并没做多路召回+重排序，因为本任务文档级别较小，如果有需要大量文档召回的朋友可以去结合v2.0的多路召回策略和rerank自行修改一下代码。  

**v5.0**  (开发-ing)
5.0版本我将使用ragas对rag系统进行评估以此验证rag的提升。


## 如何使用
首先你需要下载bge-small-zh-v1.5;bge-reranker-base;Qwen3-4B三个模型,并保存在项目路径下.  
langchain版本>0.3即可,其余不做版本要求.   
!pip install langchain  
!pip install langchain-openai  
!pip install langchain-community  
!pip install faiss-cpu  
!pip install openai  
!pip install vllm  # 可能会下载失败，可以单独create一个conda环境只下载vllm，专门用来bash start_vllm.sh使用  
!pip install chromadb  
!pip install ragas  
!pip install -U "qwen-agent[rag,code_interpreter,gui,mcp] #v3版本使用的  

