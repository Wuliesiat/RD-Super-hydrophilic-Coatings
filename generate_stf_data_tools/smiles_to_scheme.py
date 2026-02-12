# 从SMILES式子获取相关问题和scheme
from openai import OpenAI
import json
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-oYh3Xrhg8oDY2gW02c966f31C84449Ad86F9Cd9dF6E64a8d",
    base_url="https://vip.apiyi.com/v1"
)
# 添加线程锁，用于文件写入同步
file_lock = Lock()
# 添加计数器锁
counter_lock = Lock()
# 成功计数器
success_count = 0
def read_input_json(file_path):
    """从JSON文件中读取输入数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        return None
    except Exception as e:
        print(f"读取JSON文件时出错：{e}")
        return None
def fill_prompt_with_smiles(prompt_template, smiles):
    """将SMILES填充到prompt模板中"""
    return prompt_template.replace("{{SMILES}}", smiles)

def read_prompt_from_file(file_path):
    """从文件中读取提示词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        return prompt
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return None

def extract_json_from_response(response):
    """从回答中提取JSON内容"""
    try:
        # 尝试匹配 ```json ... ``` 格式
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有代码块标记，尝试直接解析整个响应
            json_str = response
        
        # 解析JSON字符串
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        print("原始响应内容：")
        print(response)
        return None
    except Exception as e:
        print(f"提取JSON时出错：{e}")
        return None

def append_json_to_file(data, file_path):
    """将JSON数据追加到文件末尾（线程安全）"""
    global success_count
    
    with file_lock:  # 添加线程锁
        try:
            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取现有的JSON数组
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                        # 确保读取的是列表
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                    except json.JSONDecodeError:
                        # 如果文件为空或格式错误，创建新列表
                        existing_data = []
            else:
                # 文件不存在，创建新列表
                existing_data = []
            
            # 追加新数据
            existing_data.append(data)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            with counter_lock:
                success_count += 1
                print(f"\n第 {success_count} 条数据已保存 (文件共有 {len(existing_data)} 条记录)")
            
            return True
            
        except Exception as e:
            print(f"保存JSON文件时出错：{e}")
            return False

def process_single_task(task_id, prompt_template, smiles_data):
    """执行单次生成任务"""
    try:
        # 提取SMILES
        smiles = smiles_data.get("smiles", "")
        if not smiles:
            print(f"[任务 {task_id}] 警告：SMILES为空")
            return False
        
        # 填充prompt
        prompt = fill_prompt_with_smiles(prompt_template, smiles)
        
        # 调用大模型生成回答
        response = generate_response(prompt)
        
        if response is None:
            print(f"[任务 {task_id}] API调用失败")
            return False
        
        # 提取JSON内容
        json_data = extract_json_from_response(response)
        
        if json_data is None:
            print(f"[任务 {task_id}] JSON提取失败")
            return False
        
        # 将原始SMILES信息也保存到结果中（可选）
        json_data["source_smiles"] = smiles
        json_data["source_deep_smiles"] = smiles_data.get("deep_smiles", "")
        
        # 追加JSON到文件
        result = append_json_to_file(json_data, "response.json")
        
        if result:
            print(f"[任务 {task_id}] 完成")
        else:
            print(f"[任务 {task_id}] 保存失败")
            
        return result
        
    except Exception as e:
        print(f"[任务 {task_id}] 异常：{e}")
        return False
    
def generate_response(prompt):
    """调用大模型生成回答"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=5000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用API时出错：{e}")
        return None

def main():
    global success_count
    
    # 从prompt.txt读取提示词模板
    prompt_template = read_prompt_from_file("prompt.txt")
    
    if prompt_template is None:
        return
    
    # 从JSON文件读取输入数据
    input_data = read_input_json("/home/ubuntu/50T/fsy/json/generate_stf_data/output_standard.json")
    
    if input_data is None:
        return
    
    # 确保input_data是列表
    if not isinstance(input_data, list):
        input_data = [input_data]
    
    total_tasks = len(input_data)

    print("\n" + "="*60)
    print("开始多线程批量生成数据")
    print("="*60)
    print(f"输入数据：{total_tasks} 条")
    print(f"线程数量：32 个（可调整）")
    print("="*60)
    
    # 重置计数器
    success_count = 0
    failed_count = 0
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用线程池执行任务
    max_workers = 32  # 可以根据API限制调整线程数
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，为每个任务传递对应的SMILES数据
        futures = {
            executor.submit(process_single_task, i+1, prompt_template, input_data[i]): i+1 
            for i in range(2300)
        }
        
        # 等待任务完成
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                if not result:
                    failed_count += 1
            except Exception as e:
                print(f"[任务 {task_id}] 执行异常：{e}")
                failed_count += 1
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印统计信息
    print("\n" + "="*60)
    print("任务完成统计")
    print("="*60)
    print(f"成功：{success_count} 条")
    print(f"失败：{failed_count} 条")
    print(f"总计：{total_tasks} 条")
    print(f"耗时：{elapsed_time:.2f} 秒")
    if total_tasks > 0:
        print(f"平均：{elapsed_time/total_tasks:.2f} 秒/条")
    print("="*60)

if __name__ == "__main__":
    main()