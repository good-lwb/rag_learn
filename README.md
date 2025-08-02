# 黄帝内经rag问答系统
![效果图1](https://github.com/good-lwb/rag_learn/blob/main/assets/0223ff4aaffd3d88cfc3141b5b168c67.png)  
![效果图2](https://github.com/good-lwb/rag_learn/blob/main/assets/ce1dac7e50b97c166bc6e2960535b0ab.png)  
本项目使用爬虫获取了皇帝内经全文以此为数据构建检索增强系统   
本项目以一个系统的多层迭代不断更新优化技术，由浅入深逐渐理解rag原理及优化技术  
在本次项目中将rag分为三步：  
1. indexing索引阶段  
2. retrive检索阶段  
3. generation生成阶段


本项目主要对这三个阶段分别展开优化

# RAG 项目版本迭代总览

| 版本   | 核心目标                          | 索引优化                          | 检索优化                          | 评估/其他                          | 框架/工具链                     |
|--------|-----------------------------------|-----------------------------------|-----------------------------------|------------------------------------|---------------------------------|
| **v1.0** | 快速体验基础RAG流程               | - 按500token分块<br>- BGE-small-zh Embedding<br>- FAISS向量库 | - 单路向量检索<br>- 无召回优化      | - 大量日志打印辅助理解             | LangChain                       |
| **v2.0** | 基础优化技术实践                  | - LangChain分块（512+128重叠）<br>- 同v1.0 Embedding/FAISS | - 双路召回（BM25+向量）<br>- 10+10→rerank Top5 | - 支持API/本地模型加载            | LangChain                       |
| **v3.0** | 企业级快速测试（平衡质量与效率）  | - 同v2.0                          | - Qwen-Rag框架内置优化<br>- 支持多轮对话 | - 本地vLLM加速推理               | Qwen-Rag + vLLM                 |
| **v4.0** | 解决长文本信息碎片化问题          | - **父文档检索器**<br>- 子块匹配→返回父块<br>- Chroma向量库+JSON持久化 | - 子块Top3→父块返回<br>- 支持API/本地Embedding | - 引入Logging<br>- 检索过程存JSON | LangChain + Chroma              |
| **v5.0** | 量化评估RAG效果                   | - 同v4.0                          | - 同v4.0                          | - **RAGAS评估**<br>（忠实度、相关性等5指标） | Ragas + 自定义模型              |
| **v6.0** | 探索LlamaIndex生态                | - LlamaIndex基础分块/索引         | - 基础检索流程                     | - 对比LangChain/Qwen-Rag          | LlamaIndex                     |

---

### **技术全景图**
```mermaid
pie
    title RAG技术覆盖度
    "分块策略" : 4 (v1.0-v4.0)
    "多路召回" : 2 (v2.0/v4.0)
    "父文档检索" : 1 (v4.0)
    "评估体系" : 1 (v5.0)
    "框架对比" : 3 (LangChain/Qwen/Llama)


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

*索引*：使用langchain.retrievers提供的父文档检索器（ParentDocumentRetriever）构建Chroma本地向量数据库。  
  在这部分在由于ParentDocumentRetriever不支持持久化保存父文档信息我们单独构建json将父文档chunk保存，再通过json在rag检索时进行加载，使得检索精度得到极大的提升。  
  同时，支持两中加载embedding模型的方式（api/本地），通过参数控制方便使用。  
  除此之外提供test_chroma_db.py文件支持对本地chroma向量数据库进行测试。  

*检索*：父文档检索器（检索top_k=3个相似度最高的子文档，并根据重构的json返回对应的父文档片段）  
  4.0版本中将llm的本地化加载合并为同一main.py中并通过指定remote/local参数进行配置使用阿里云api加载或本地部署的vllm。  
*日志*：在4.0中引入了logging功能，并将检索信息通过json文件进行保存，方便后续对rag检索能进行评估。  
在这部分我并没做多路召回+重排序，因为本任务文档级别较小，如果有需要大量文档召回的朋友可以去结合v2.0的多路召回策略和rerank自行修改一下代码。  



**v5.0**    
5.0版本使用ragas对rag系统进行评估以此验证在4.0版本rag的提升（其他版本测试代码这里不给出了，我已经将ragas评估函数封装，直接传参进去测试其他版本不难）。  
衡量一个rag系统的主要参数有如下五类：  
忠实度(faithfulness)：衡量了生成的答案(answer)与给定上下文(context)的事实一致性。它是根据answer和检索到的context计算得出的。并将计算结果缩放到 (0,1) 范围且越高越好。  
答案相关性(Answer relevancy)：重点评估生成的答案(answer)与用户问题(question)之间相关程度。不完整或包含冗余信息的答案将获得较低分数。该指标是通过计算question和answer获得的，它的取值范围在 0 到 1 之间，其中分数越高表示相关性越好。  
上下文精度(Context precision)：评估所有在上下文(contexts)中呈现的与基本事实(ground-truth)相关的条目是否排名较高。理想情况下，所有相关文档块(chunks)必须出现在顶层。该指标使用question和计算contexts，值范围在 0 到 1 之间，其中分数越高表示精度越高。  
上下文召回率(Context recall)：衡量检索到的上下文(Context)与人类提供的真实答案(ground truth)的一致程度。它是根据ground truth和检索到的Context计算出来的，取值范围在 0 到 1 之间，值越高表示性能越好。  
上下文相关性(Context relevancy)：该指标衡量检索到的上下文(Context)的相关性，根据用户问题(question)和上下文(Context)计算得到，并且取值范围在 (0, 1)之间，值越高表示相关性越好。理想情况下，检索到的Context应只包含解答question的信息。   
最终测试结果如下所示（因为ragas默认需要使用opanai的ai，我这里我没有就硬传的自己的模型，虽然有结果但是不知道为什么有几个指标都是零，这个我后续在研究补充）：   
![测试图](https://github.com/good-lwb/rag_learn/blob/main/assets/evaluate.png)


**v6.0**   
6.0版本使用llama_index完成了黄帝内经系统的开发，但只是使用了最基础的开发框架，没有进行优化。  
llama_index相较于langchain生态更加完善，集成的编码、检索优化等手段更加方便，基本都是开箱即用（内部包支持），不像langchain手写的多。  
这里没给出基于llama_index的优化，llama_index生态文档很完善可以看文档，基本优化都很简单官方集成了很多的优化而且都是换个参数改个函数就能实现。  



**心得**
在这个项目中，我一共使用了三个框架对黄帝内经rag进行开发（langchain、llama_index、Qwen-rag）  
说一下我个人感觉：   
Qwen-rag的话优化做得很好，而且上手很简单，官方抽象的很好基本就是把自己的doc扔进去就可以使用而且有不错的效果，适合想要快速体验rag给llm带来的提升效果的同学，或者项目着急出个实体给客户看一眼也可以使用。   
llama_index对于数据库构建、索引，包括生成其实都有很强的抽象程度，而且用起来真的很方便，感觉很多人都在吹langchain（包括我自己），但实际用下来rag感觉还是llama_index更好一些。  
langchain怎么说呢又爱又恨，真的很多都要自己手写，比如数据库检索之类的，但是你说他不行它prompt模板有很好用，而且他还有自己的一套生态比如LCEL（管道符执行，非常有意思），而且她对模型调用、agent、工作流的支持也都不是llama_index能做到的。  


综上没有说什么好什么不好，干这行你就都得会，都得学，而且其实现在你看langchain和llama_index的文档可以很明显的感觉到，他俩越来越像了，就比如文档加载器都要一摸一样了，后面他们各自优化其实也会更相似。

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
!pip install llama-index  # 安装llama_index必须新create虚拟环境，和langchain不兼容  
!pip install llama-index-embeddings-huggingface  
!pip install llama-index-llms-huggingface  



