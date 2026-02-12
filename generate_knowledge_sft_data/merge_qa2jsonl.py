# import os
# import json
# import random

# # 设定目录路径
# directory_path = '/home/ubuntu/50T/fsy/wl/task1/task1-qa'  # 替换为你的目录
# output_file_1 = '/home/ubuntu/50T/fsy/wl/task1/task1_train_dataset_new.jsonl'    # 输出的第一个 jsonl 文件名
# output_file_2 = '/home/ubuntu/50T/fsy/wl/task1/task1_val_dataset_new.jsonl'    # 输出的第二个 jsonl 文件名

# # 获取目录下所有 JSON 文件
# json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
# random.shuffle(json_files)  # 打乱文件顺序

# # 随机选择 30 个文件
# num_random_files = 30
# if len(json_files) < num_random_files:
#     num_random_files = len(json_files)

# random_files = json_files[:num_random_files]
# remaining_files = json_files[num_random_files:]

# # 保存到 jsonl 文件
# def save_to_jsonl(file_list, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for json_file in file_list:
#             with open(os.path.join(directory_path, json_file), 'r', encoding='utf-8') as json_f:
#                 data = json.load(json_f)
#                 data_done = {"messages": [{"role": "system", "content": "You are a seasoned professor in the field of materials science, with primary research focusing on the monomer synthesis of hydrophilic polymers."}, {"role": "user", "content": data["design_question"]}, {"role": "assistant", "content": data["design_answer"]}]}
#                 # data_done = {"system":"You are a seasoned professor in the field of materials science, with primary research focusing on the monomer synthesis of hydrophilic polymers.", "conversation":[{"prompt":data["design_question"],"response":data["design_answer"]}]}
#                 f.write(json.dumps(data_done) + '\n')

# # 将文件保存到对应的 jsonl 文件
# save_to_jsonl(remaining_files, output_file_1)
# save_to_jsonl(random_files, output_file_2)

# print(f'已将 {len(remaining_files)} 个文件保存到 {output_file_1}')
# print(f'已将 {len(random_files)} 个文件保存到 {output_file_2}')


import json
import random

# 假设这是你的两个输入 JSONL 文件
input_file_1 = '/home/ubuntu/50T/fsy/wl/task1_val_dataset_new.jsonl'  # 第一个 JSONL 文件名
input_file_2 = '/home/ubuntu/50T/fsy/wl/task2_val_dataset_new.jsonl'  # 第二个 JSONL 文件名
output_file = '/home/ubuntu/50T/fsy/wl/val_dataset.jsonl'    # 合并后输出的 JSONL 文件名

# 读取 JSONL 文件并存储到列表中
def read_jsonl(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 读取两个 JSONL 文件
data1 = read_jsonl(input_file_1)
data2 = read_jsonl(input_file_2)

# 合并数据
merged_data = data1 + data2

# 随机打乱合并后的数据
random.shuffle(merged_data)

# 将打乱后的数据写入新的 JSONL 文件
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in merged_data:
        f.write(json.dumps(entry) + '\n')

print(f'已将 {len(merged_data)} 条记录合并并打乱，保存到 {output_file}')
