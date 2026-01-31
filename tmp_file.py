
import json
import os


tasks = {
    "medqa": "medqa_source_test.json",
    "mmlupro": "mmlupro_test.json",
    "pubmedqa": "pubmedqa_labeled_train.json",
    "factor": "factor_mc.json",
    "gpqa": "gpqa_train.json",
    "arlsat": "arlsat_all.json",
    "logiqa": "logiqa_mrc.json"
}

merged_data = []

for task_name, file_name in tasks.items():
    file_path = os.path.join(task_name, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 如果数据是列表，直接处理；如果是字典（如 MMLU-Pro 常见结构），转为列表
            if isinstance(data, dict):
                # 尝试获取常见的数据键，如果都没有，就把整个字典包装进列表
                data_list = data.get('data', data.get('test', [data]))
            else:
                data_list = data
            
            # 为每条数据增加 task 标记，方便以后按 task 统计结果
            for item in data_list:
                if isinstance(item, dict):
                    item['task_source'] = task_name
                merged_data.append(item)
        print(f"✅ 已加载 {task_name}: {len(data_list)} 条数据")
    else:
        print(f"❌ 未找到文件: {file_path}")

# 保存合并后的文件
with open('benchmark.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"\n🚀 合并完成！总计 {len(merged_data)} 条数据，已保存为 benchmark.json")