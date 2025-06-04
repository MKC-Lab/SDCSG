#!/bin/bash

# 脚本开始
echo "开始运行任务：Mistral_7B_Base - 2Positive"
python main.py --model_name Mistral_7B_Base --task_name 2Positive

echo "任务完成：Mistral_7B_Base - 2Positive"

echo "开始运行任务：Qwen2_7B_Base - 2Negative"
python main.py --model_name Qwen2_7B_Base --task_name 2Negative

echo "开始运行任务：llama3_8B_Base - 2Positive"
python main.py --model_name llama3_8B_Base --task_name 2Positive

echo "全部任务完成！"
