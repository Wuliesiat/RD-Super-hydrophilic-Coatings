import json
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

class ThreadSafeResultCollector:
    """线程安全的结果收集器"""
    def __init__(self):
        self.results = []
        self.lock = Lock()
    
    def add_result(self, result):
        with self.lock:
            self.results.append(result)
    
    def get_results(self):
        return self.results
def process_single_data(idx, input_data, client, result_collector):
    """
    处理单条数据的函数（线程工作单元）
    
    Args:
        idx: 数据索引
        input_data: 单条输入数据
        client: OpenAI客户端
        result_collector: 结果收集器
    """
    try:
        print(f"[线程 {threading.current_thread().name}] 开始处理第 {idx} 条数据...")
        
        # 创建prompt
        prompt = create_prompt(input_data)
        
        # 调用大模型
        llm_response = call_llm(prompt, client)
        
        # 构建结果
        result = {
            "index": idx,  # 添加索引以便排序
            "question": input_data.get('user_query', ''),
            "scheme": input_data.get('tool_schema', {}),
            "answer": llm_response
        }
        
        # 线程安全地添加结果
        result_collector.add_result(result)
        
        print(f"[线程 {threading.current_thread().name}] 完成第 {idx} 条数据")
        
    except Exception as e:
        print(f"[错误] 处理第 {idx} 条数据时出错: {str(e)}")
        # 即使出错也记录结果
        error_result = {
            "index": idx,
            "question": input_data.get('user_query', ''),
            "scheme": input_data.get('tool_schema', {}),
            "answer": f"Error: {str(e)}"
        }
        result_collector.add_result(error_result)    
def read_json_file(file_path):
    """读取JSON文件，支持单条或多条数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是单条数据（字典），转换为列表
    if isinstance(data, dict):
        return [data]
    # 如果已经是列表，直接返回
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON文件格式不正确，应为字典或列表")
def create_prompt(data):
    """根据JSON数据创建prompt"""
    tool_schema = data.get('tool_schema', {})
    user_query = data.get('user_query', '')
    
    # 将tool_schema转换为格式化的JSON字符串
    tool_schema_str = json.dumps(tool_schema, indent=2, ensure_ascii=False)
    
    # 构建完整的prompt
    prompt = f"""
你是一个专业的问答助手，擅长利用已有工具解决用户提出的复杂问题。你需要根据用户的问题和提供的工具定义（Scheme），进行深入的逻辑分析，并生成回复。

# Instructions
1. **意图识别与执行**：
   - 不需要输出思考过程。直接在内部评估用户意图和现有信息完整性。
   - 如果现有知识缺失，无法准确回答问题时，必须调用工具，**严禁凭空捏造**。

2. **工具调用规则**：
   - 当需要外部辅助时，直接生成工具调用代码，不要犹豫。
   - 遇到不确定的内容，**绝对禁止胡编乱造**，必须通过工具获取准确信息。
   - 工具调用必须使用 XML 标签包裹：`<tool_call>...</tool_call>`。
   - `<tool_call>` 内部必须是一个合法的 JSON 对象，包含 `name` 和 `arguments` 两个字段。
   - **关键格式要求**：`arguments` 字段的值必须是一个**经过转义的 JSON 字符串**。
   - 格式示例：
     <tool_call>
     {{"name": "search_func", "arguments": "{{\\"key\\": \\"value\\", \\"param\\": \\"data\\"}}"}}
     </tool_call>

3. **回复逻辑**：
   - **调用工具时**：充分说明即将进行的操作，紧接着输出 <tool_call> 块。
   - **直接回答时**：直接输出最终答案，确保逻辑清晰、准确。
   
# Tool Scheme (Available Tools)
以下你可以调用的工具定义（OpenAPI JSON格式）：
{tool_schema_str}

# User Query
{user_query}
"""
    
    return prompt

def call_llm(prompt, client):
    """调用大模型API"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或使用其他模型
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手，擅长利用已有工具解决用户提出的复杂问题。你需要根据用户的问题和提供的工具定义（Scheme），进行深入的逻辑分析，并生成回复。"},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=5000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def save_results(output_path, results):
    """保存所有结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"所有结果已保存到: {output_path}")

def main():
    # 配置参数
    input_file = "/home/ubuntu/50T/fsy/json/generate_stf_data/input.json"
    output_file = "output.json"
    max_workers = 32  # *** 新增：设置线程数 ***
    
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="",
        base_url=""
    )
    
    # 读取输入JSON
    print("正在读取输入文件...")
    input_data_list = read_json_file(input_file)
    print(f"共读取到 {len(input_data_list)} 条数据")
    
    # *** 新增：创建结果收集器 ***
    result_collector = ThreadSafeResultCollector()
    
    # *** 修改：使用线程池处理 ***
    print(f"\n开始多线程处理（线程数: {max_workers}）...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                process_single_data, 
                idx, 
                input_data, 
                client, 
                result_collector
            ): idx 
            for idx, input_data in enumerate(input_data_list, 1)
        }
        
        # 等待所有任务完成
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()  # 获取结果（如果有异常会在这里抛出）
            except Exception as e:
                print(f"[严重错误] 任务 {idx} 执行失败: {str(e)}")
    
    # *** 新增：按索引排序结果 ***
    all_results = sorted(result_collector.get_results(), key=lambda x: x['index'])
    
    # *** 修改：移除索引字段（可选）***
    for result in all_results:
        result.pop('index', None)
    
    # 保存所有结果
    print(f"\n{'='*50}")
    print("正在保存所有结果...")
    save_results(output_file, all_results)
    
    print("处理完成！")

if __name__ == "__main__":
    main()