# 将带有工具调用的结果返回给大模型
import json
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time

class ProgressTracker:
    """线程安全的进度跟踪器"""
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.lock = threading.Lock()
    
    def update(self):
        with self.lock:
            self.completed += 1
            print(f"Progress: {self.completed}/{self.total} items processed")

def process_single_item(item, idx, client, prompt_template, progress_tracker):
    """
    处理单个数据项
    
    Args:
        item: 要处理的数据项
        idx: 项目索引
        client: OpenAI客户端
        prompt_template: prompt模板
        progress_tracker: 进度跟踪器
    
    Returns:
        处理后的数据项
    """
    print(f"Processing item {idx + 1}...")
    
    # 提取question和answer
    question = item.get('question', '')
    answer = item.get('answer', '')
    
    # 填充prompt模板
    filled_prompt = prompt_template.replace('{{USER_QUESTION}}', question)
    filled_prompt = filled_prompt.replace('{{TOOL_OUTPUTS}}', answer)
    
    try:
        # 调用大模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": filled_prompt}
            ],
            temperature=0.8,
        )
        
        # 获取模型输出
        new_answer = response.choices[0].message.content
        
        # 构建输出数据
        output_item = item.copy()
        output_item['answer_tool'] = answer
        output_item['answer'] = new_answer
        
        print(f"Item {idx + 1} processed successfully.")
        
    except Exception as e:
        print(f"Error processing item {idx + 1}: {str(e)}")
        output_item = item.copy()
        output_item['answer_tool'] = answer
        output_item['answer'] = f"ERROR: {str(e)}"
        output_item['processing_error'] = True
    
    finally:
        progress_tracker.update()
    
    return idx, output_item

def process_json_with_llm(input_json_path, output_json_path, prompt_file_path, max_workers=5):
    """
    使用多线程处理JSON文件中的数据，使用大模型生成新的answer
    
    Args:
        input_json_path: 输入JSON文件路径
        output_json_path: 输出JSON文件路径
        prompt_file_path: prompt模板文件路径
        max_workers: 最大线程数（默认5）
    """
    
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="",
        base_url=""
    )
    
    # 读取prompt模板
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # 读取输入JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 如果输入是单个对象而不是列表，转换为列表
    if isinstance(input_data, dict):
        input_data = [input_data]
    
    # 初始化进度跟踪器
    progress_tracker = ProgressTracker(len(input_data))
    
    # 使用字典存储结果，保持原始顺序
    results_dict = {}
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用线程池处理数据
    print(f"Starting processing with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(
                process_single_item, 
                item, 
                idx, 
                client, 
                prompt_template, 
                progress_tracker
            ): idx 
            for idx, item in enumerate(input_data)
        }
        
        # 收集结果
        for future in as_completed(future_to_idx):
            try:
                idx, output_item = future.result()
                results_dict[idx] = output_item
            except Exception as e:
                idx = future_to_idx[future]
                print(f"Unexpected error for item {idx + 1}: {str(e)}")
                # 保存错误结果
                error_item = input_data[idx].copy()
                error_item['answer_tool'] = error_item.get('answer', '')
                error_item['answer'] = f"UNEXPECTED ERROR: {str(e)}"
                error_item['processing_error'] = True
                results_dict[idx] = error_item
    
    # 按原始顺序整理结果
    output_data = [results_dict[idx] for idx in sorted(results_dict.keys())]
    
    # 保存结果到输出JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    
    print(f"\nProcessing complete! Results saved to {output_json_path}")
    print(f"Total items processed: {len(output_data)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per item: {elapsed_time/len(output_data):.2f} seconds")

if __name__ == "__main__":
    # 配置文件路径
    input_json_path = "/home/ubuntu/50T/fsy/json/generate_stf_data/response.json"
    output_json_path = "response.json"
    prompt_file_path = "prompt.txt"
    
    # 设置线程数
    max_workers = 32  
    
    # 执行处理
    process_json_with_llm(
        input_json_path, 
        output_json_path, 
        prompt_file_path,
        max_workers=max_workers
    )