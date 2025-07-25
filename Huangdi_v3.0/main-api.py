import os
from qwen_agent.agents import Assistant

# 配置 LLM
llm_cfg = {
    'model': 'qwen-max',  # 或其他 Qwen 模型
    'model_server': 'dashscope',
    'api_key': os.getenv("DASHSCOPE_API_KEY"),  # 确保环境变量已设置，可以暂存在自己的环境变量里export DASHSCOPE_API_KEY='sk-xxxx'
    'generate_cfg': {'top_p': 0.8}
}

def get_file_list(folder_path):
    # 初始化文件列表
    file_list = []
    
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 将文件路径添加到列表中
            file_list.append(file_path)
    return file_list

# 初始化 Assistant
bot = Assistant(
    llm=llm_cfg,
    system_message="你是一个专业的中医问答小助手，请根据提供的文件回答问题，同时注意不要重复输出，可以适当的输出部分检索到的片段内容，以此来增加说服力。",
    function_list=[],  # 可以添加工具（如搜索、计算等）
    files=get_file_list('./data')  # 加载文档
)

# 初始化对话历史
messages = []

def chat_with_bot(query):
    global messages
    # 添加用户输入到对话历史
    messages.append({'role': 'user', 'content': query})
    
    print("\n用户提问:", query)
    print("\nAI回复:", end=" ")
    
    # 运行 Agent 并获取流式响应
    current_index = 0
    full_response = ""
    for response in bot.run(messages=messages):
        # 流式输出
        current_response = response[0]['content'][current_index:]
        current_index = len(response[0]['content'])
        print(current_response, end='')
        full_response += current_response

    # 将 AI 回复加入对话历史
    messages.append({'role': 'assistant', 'content': full_response})
    return full_response

# 示例多轮对话
if __name__ == "__main__":
    while True:
        user_input = input("\n\n请输入您的问题（输入 'exit' 退出）: ").strip()
        if user_input.lower() == 'exit':
            break
        chat_with_bot(user_input)