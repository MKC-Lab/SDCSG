# import json
# import re

# input_file = '/root/autodl-tmp/DATG/data/test_stance.jsonl'         # 原始 JSON 文件路径
# output_file = '/root/autodl-tmp/DATG/data/new_test.jsonl'  # 输出 JSONL 文件路径

# with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
#     for line in fin:
#         data = json.loads(line)
#         original_prompt = data.get('prompt', '')

#         # 提取 stance（favor 或 against）
#         stance_match = re.search(r'stance:\s*(favor|against)', original_prompt, re.IGNORECASE)
#         stance = stance_match.group(1).lower() if stance_match else ''

#         # 提取 User post 的文本
#         if "User post: " in original_prompt:
#             user_post = original_prompt.split("User post: ")[-1].strip()
#         else:
#             user_post = ''

#         # 更新字段
#         data['prompt'] = user_post       # 用新 prompt 替换原 prompt 内容
#         data['stance'] = stance          # 添加 stance 字段

#         fout.write(json.dumps(data, ensure_ascii=False) + '\n')

# print(f"✅ 已提取 stance 和 prompt，结果保存为：{output_file}")



import json

input_file = '/root/autodl-tmp/DATG/data/new_test.jsonl'     # 已经提取 stance 的文件
favor_file = '/root/autodl-tmp/DATG/data/test_favor.jsonl'
against_file = '/root/autodl-tmp/DATG/data/test_against.jsonl'

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(favor_file, 'w', encoding='utf-8') as ffavor, \
     open(against_file, 'w', encoding='utf-8') as fagainst:

    for line in fin:
        data = json.loads(line)
        stance = data.get('stance', '').lower()

        if stance == 'favor':
            ffavor.write(json.dumps(data, ensure_ascii=False) + '\n')
        elif stance == 'against':
            fagainst.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"✅ 拆分完成：")
print(f" - favor 数据保存为：{favor_file}")
print(f" - against 数据保存为：{against_file}")
