from openai import OpenAI
from pathlib import Path
import os
import re
import json
import glob
import tqdm
from multiprocessing import Pool
from functools import partial
from collections import Counter

API_KEY = "sk-oYh3Xrhg8oDY2gW02c966f31C84449Ad86F9Cd9dF6E64a8d"
BASE_URL = "https://vip.apiyi.com/v1"
MODEL_GPT = "gpt-4o-mini"

# 确保输出为标准json格式字符串
def comfirm_json_string_gpt(json_string):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You will read a <string>, please fix this string into a string that can be parsed by json.loads.

        Note:
        1. No descriptive text is required.
        2. Don't use markdown syntax.

        The <string>: {json_string}
    """

    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": "You are an assistant who is proficient in material synthesis."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# 确保输出为标准json格式字符串
def comfirm_json_string(json_string):
    json_string = re.sub(r'[“”]', '"', json_string)
    json_string = re.sub(r'\\', r'\\\\', json_string)
    json_string = re.sub(r'\\"', r'\"', json_string)
    json_string = json_string.replace("\n", "").replace("\r", "")
    # 去掉 Markdown 的语法包裹
    if json_string.startswith("```json"):
        json_string = json_string.strip("`json\n")
    json_string = json_string.strip('`\n')

    return json_string

# 文本分割
def split_by_heading(markdown_text, heading_level='#'):
    # `heading_level` could be '#', '##', '###', etc.
    pattern = r'(?=\n{})'.format(re.escape(heading_level))
    
    # 使用正则表达式进行切割，以包含标题的内容
    split_texts = re.split(pattern, markdown_text)
    
    # 去除空白的块
    return [block.strip() for block in split_texts if block.strip()]

# 文本段分类
def segment_classification(text_split):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You will read a text segment about hydrophilic polymers. Please analyze which part of a paper this segment belongs to and give your classification result. The categories you can only choose are as follows:
        1. Abstract
        2. Introduction
        3. Materials and methods
        4. Results and discussion
        5. Conclusions
        6. References

        Please output the result using the following format:
        Category: Abstract/Introduction/Materials and methods/Results and discussion/Conclusions/References 

        Text segment as follows: {text_split}
    """

    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": "You are an expert in interdisciplinary research across fields such as materials chemistry, polymer science, biomaterials engineering, and interface and surface science."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# 处理单个md文件
def process_file(md_path, output_dir):
    chunks = []
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # 将文本按heading分割
    content_splits = split_by_heading(md_content)

    id = 0
    for content_split in content_splits:
        id += 1
        chunk = {}
        result = segment_classification(content_split)
        chunk["id"] = id
        chunk["chunk"] = content_split
        chunk["category"] = result[9:]
        chunks.append(chunk)

    output_path = os.path.join(output_dir, os.path.basename(md_path).replace('.md', '.json'))
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(chunks, json_file, ensure_ascii=False, indent=4)

# 获取已经处理过的md
def chunk_done(json_dir):
    jsons = os.listdir(json_dir)
    json_names = [json_name.replace('.json', '') for json_name in jsons]
    return json_names

# 将文本段分割分类并保存为json
def md_segment():
    md_paths = glob.glob("/home/ubuntu/50T/fsy/wl/articles/mds/**/*.md", recursive=True)
    print("md文件数量：", len(md_paths))
    # 过滤已经过处理的文件
    output_dir = "/home/ubuntu/50T/fsy/wl/task1-chunks"
    json_names = chunk_done(output_dir)
    md_paths = [md_path for md_path in md_paths if os.path.basename(md_path).replace(".md", "") not in json_names]
    print("过滤后md文件数量：", len(md_paths))
    
    for path in tqdm.tqdm(md_paths):
        try:
            process_file(path, output_dir)
        except Exception as e:
            print(f"处理 {path} 时出错: {e}")

    # # 设置多进程池
    # pool = Pool(processes=32)

    # process_func = partial(process_file, output_dir=output_dir)

    # # imap_unordered 将逐步从 md_paths 传给 process_func 进行并行处理
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, md_paths), total=len(md_paths)):
    #     pass

    # pool.close()
    # pool.join()

# 提取分子做亲水性聚合物的单体结构及其有助于亲水性的相应官能团
def get_function_groups(text):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You will read a text excerpt about the synthesis of hydrophilic polymers. Please extract all information regarding the monomer structures used for synthesizing hydrophilic polymers, including:
        1. Information about the functional groups that enhance the hydrophilicity of the corresponding polymers.
        2. Explanations of how these functional groups enhance interactions with water.

        Note:
        1. The information you extract must come from the text excerpt(example not included), and fabrication of information is strictly prohibited.
        2. Don't use markdown syntax.
        3. If no relevant information is extracted, return the format with the "content" field left empty.

        Please output the result using the following format:
        {{
            "content": "a single complete sentence containing all the required information",
        }}

        The text except: {text}
    """

    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
             {"role": "system", "content": "You are an expert in developing hydrophilic polymers for applications such as biomedical hydrogels or water filtration membranes."},
            # {"role": "system", "content": "You are an expert in researching surface modification of black phosphorus."},
            {"role": "user", "content": prompt}
        ]
    )

    return  response.choices[0].message.content

# 提取实验方案
def extract_info(chunks_path):
    with open(chunks_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    protocol_dict = {"content" : ""}   # 存放最终输出
    for chunk in chunks:
        chunk_text = chunk['chunk']
        category = chunk['category']
        try:
            # 提取分子做亲水性聚合物的单体结构及其有助于亲水性的相应官能团
            if category == ' Introduction' or category == ' Materials and methods' or category == 'Results and discussion':
                intermediate_result = get_function_groups(chunk_text)
                print(intermediate_result)
                intermediate_result = comfirm_json_string(intermediate_result)
                try:
                    result_protocol = json.loads(intermediate_result)
                except json.JSONDecodeError as e:
                    # 修复json字符串(gpt)
                    escaped_protocol = comfirm_json_string_gpt(intermediate_result)
                    try:
                        result_protocol = json.loads(escaped_protocol)
                    except Exception as e:
                        print(e)
                        print(escaped_protocol)
                        return
                if result_protocol["content"] == "":
                    continue
                if isinstance(result_protocol, dict):
                    protocol_dict["content"] += result_protocol["content"]
                else:
                    print("result_content不是一个字典")
                    print(result_protocol)
                    return
        except Exception as e:
            print(e)
            return

    output_path = os.path.join(output_dir, os.path.basename(chunks_path))
    with open(output_path, 'w') as json_file:
        json.dump(protocol_dict, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    chunks_dir = "/home/ubuntu/50T/fsy/wl/task1-chunks"
    paths = [os.path.join(chunks_dir, path) for path in os.listdir(chunks_dir)]
    print("chunks文件数量：", len(paths))

    # 过滤已处理的文件
    output_dir = "/home/ubuntu/50T/fsy/wl/task1-paper-info"
    proccessed_files = [path for path in os.listdir(output_dir)]
    paths = [path for path in paths if os.path.basename(path) not in proccessed_files]
    print("过滤后chunks文件数量：", len(paths))
    
    # step1
    # md_segment()
    
    # step2
    for path in tqdm.tqdm(paths):
        try:
            extract_info(path)
        except Exception as e:
            print(f"处理 {path} 时出错: {e}")
    
    
    # 设置多进程池
    # pool = Pool(processes=32)

    # process_func = partial(extract_info)

    # # imap_unordered 将逐步从 md_paths 传给 process_func 进行并行处理
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, paths), total=len(paths)):
    #     pass

    # pool.close()
    # pool.join()
