import json
import random

input_file_1 = '/home/ubuntu/50T/fsy/wl/task1_val_dataset_new.jsonl'  # 第一个 JSONL 文件名
input_file_2 = '/home/ubuntu/50T/fsy/wl/task2_val_dataset_new.jsonl'  # 第二个 JSONL 文件名
output_file = '/home/ubuntu/50T/fsy/wl/val_dataset.jsonl'    # 合并后输出的 JSONL 文件名

# 读取 JSON 文件并存储到列表中
def read_jsonl(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 读取两个 JSON 文件
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
