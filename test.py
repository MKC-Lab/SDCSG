import argparse
import os
from method.datg.datg import ModelResourceManager, DatgTextGenerator
from method.sdcpag.sdcpag import SDCPAGTextGenerator
from tqdm import tqdm
import pandas as pd
from config import MODEL_PATHS, TASK_CONFIGURATIONS, GENERATION_CONFIGS

NUM_SENTENCES = 10
NODES_NUM = 5

def run_sdcpag_experiment(task_name, model_name, sample_size=None):
    """
    运行SD-CPAG方法实验并将结果保存为JSONL文件
    
    Args:
        task_name: 任务类型 ("2Positive" 或 "2Negative")
        model_name: 模型名称
        sample_size: 可选的样本数量限制，用于快速测试
    """
    print(f"======= 开始SD-CPAG实验: {task_name} 使用 {model_name} =======")
    
    # 检查任务类型
    if task_name not in ["2Positive", "2Negative"]:
        raise ValueError(f"不支持的任务: {task_name}。只支持 '2Positive' 和 '2Negative'")
    
    # 获取模型路径和任务配置
    model_path = MODEL_PATHS[model_name]
    task_config = TASK_CONFIGURATIONS[task_name]
    
    # 加载资源管理器
    print("正在加载模型...")
    resource_manager = ModelResourceManager(
        model_path=model_path,
        classifier_path=task_config["classifier_path"],
        base_model_path=task_config["base_model_path"]
    )
    
    # 初始化SD-CPAG文本生成器
    sdcpag_text_generator = SDCPAGTextGenerator(
        resource_manager, 
        num_sentences=NUM_SENTENCES, 
        nodes_num=NODES_NUM
    )
    
    # 目标立场
    target_stance = "positive" if task_name == "2Positive" else "negative"
    
    # 处理每个数据集
    for data_test, data_path in TASK_CONFIGURATIONS[task_name]["data_path"].items():
        print(f"\n处理数据集: {data_test} (任务: {task_name})")
        
        # 加载数据集
        test_data = pd.read_json(data_path, lines=True)
        
        # 可选: 限制样本大小进行快速测试
        if sample_size and sample_size < len(test_data):
            test_data = test_data.head(sample_size)
            print(f"限制样本数量为 {sample_size} 进行测试")
            
        # 创建用于存储结果的列
        result_column = f'SDCPAG_{model_name}'
        
        # 应用SD-CPAG方法
        tqdm.pandas(desc=f"使用SD-CPAG生成文本")
        test_data[result_column] = test_data.index.to_series().progress_apply(
            lambda idx: generate_with_error_handling(
                sdcpag_text_generator, 
                test_data.loc[idx, 'prompt'],
                GENERATION_CONFIGS,
                target_stance
            )
        )
        
        # 保存结果
        result_dir = task_config['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        # 保存为JSONL文件
        result_file = os.path.join(result_dir, f"sdcpag_{model_name}_{data_test}_{task_name}.jsonl")
        test_data.to_json(result_file, orient='records', lines=True, force_ascii=False)
        print(f"结果已保存至JSONL文件: {result_file}")

def generate_with_error_handling(generator, prompt, configs, target_stance):
    """
    带错误处理的文本生成函数
    
    Args:
        generator: SD-CPAG文本生成器
        prompt: 输入提示
        configs: 生成配置
        target_stance: 目标立场
        
    Returns:
        str: 生成的文本或错误信息
    """
    try:
        return generator.generate_sdcpag_text(prompt, configs, target_stance)
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行SD-CPAG方法实验')
    parser.add_argument("--model_name", type=str, choices=list(MODEL_PATHS.keys()), 
                        default="phi2_3B_Base", help="可用模型名称")
    parser.add_argument("--task_name", type=str, choices=["2Positive", "2Negative"], 
                        default="2Positive", help="任务类型")
    parser.add_argument("--sample_size", type=int, default=None, 
                        help="用于测试的样本数量限制 (不设置则使用全部样本)")
    
    args = parser.parse_args()
    
    run_sdcpag_experiment(args.task_name, args.model_name, args.sample_size)