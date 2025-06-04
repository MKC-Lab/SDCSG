import argparse
import os
from method.datg.datg import ModelResourceManager, DatgTextGenerator
from method.fudge.fudge import FudgeTextGenerator
from method.preadd.preadd import PreaddTextGenerator
from method.sdcpag import SDCPAGTextGenerator
from tqdm import tqdm
import pandas as pd
from config import MODEL_PATHS, TASK_CONFIGURATIONS, GENERATION_CONFIGS
from method.sdcpag.semantic_only_sdcpag import SemanticOnlyTextGenerator

NUM_SENTENCES = 20
NODES_NUM = 5
BOOST_VALUE = 4.0
AVOID_VALUE = 6.0
FUDGE_ALPHA = 0.5
PREADD_STRENGTH = 1.0

def main(task_name, model_name):
    # 检查任务类型，只支持积极或消极立场
    if task_name not in ["2Positive", "2Negative"]:
        raise ValueError(f"Unsupported task name: {task_name}. Only '2Positive' and '2Negative' are supported.")
    
    # 获取特定模型路径和数据配置
    model_path = MODEL_PATHS[model_name]
    task_config = TASK_CONFIGURATIONS[task_name]

    # 加载资源管理器
    resource_manager = ModelResourceManager(
        model_path=model_path,
        classifier_path=task_config["classifier_path"],
        base_model_path=task_config["base_model_path"]
    )

    # 初始化各种文本生成器
    datg_text_generator = DatgTextGenerator(resource_manager, num_sentences=NUM_SENTENCES, nodes_num=NODES_NUM)
    fudge_text_generator = FudgeTextGenerator(resource_manager)
    preadd_text_generator = PreaddTextGenerator(resource_manager)
    sdcpag_text_generator = SDCPAGTextGenerator(resource_manager, num_sentences=NUM_SENTENCES, nodes_num=NODES_NUM)
    semantic_only_generator = SemanticOnlyTextGenerator(resource_manager)
    # 目标立场
    target_stance = "positive" if task_name == "2Positive" else "negative"
    
    # 迭代处理每个数据集
    for data_test, data_path in TASK_CONFIGURATIONS[task_name]["data_path"].items():
        print(f"Processing dataset: {data_test} for {task_name} task")

        # 加载数据集
        test_data = pd.read_json(data_path, lines=True)

        # # 生成初始文本
        # tqdm.pandas(desc=f"Generating multiple sentences for {model_name} on {data_test}")
        # generated_sentences_dict = {}
        # for idx, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Generating text"):
        #     prompt = row['target']
        #     generated_texts = datg_text_generator.generate_texts(prompt, GENERATION_CONFIGS, num_return_sequences=NUM_SENTENCES)
        #     generated_sentences_dict[idx] = generated_texts

        # # 处理生成的文本并提取重要节点
        # important_nodes_positive_dict = {}
        # important_nodes_negative_dict = {}
        # for idx in tqdm(test_data.index, desc="Extracting important nodes"):
        #     initial_sentences = generated_sentences_dict[idx]
        #     positive_graph, negative_graph = datg_text_generator.graph_processor.process_sentences_dual_graph(initial_sentences, task=task_name)
        #     important_nodes_positive = datg_text_generator.graph_processor.find_important_nodes(positive_graph, NODES_NUM)
        #     important_nodes_negative = datg_text_generator.graph_processor.find_important_nodes(negative_graph, NODES_NUM)
        #     important_nodes_positive_dict[idx] = important_nodes_positive
        #     important_nodes_negative_dict[idx] = important_nodes_negative

        # # 应用SD-CPAG方法
        # tqdm.pandas(desc=f"Processing SDCPAG for {model_name} on {data_test}")
        # test_data[f'SDCPAG_{model_name}'] = test_data.index.to_series().progress_apply(
        #     lambda idx: sdcpag_text_generator.generate_sdcpag_text(
        #         test_data.loc[idx, 'target'],
        #         GENERATION_CONFIGS,
        #         target_stance
        #     )
        # )
        
        # 应用仅语义级的SD-CPAG方法（消融实验）
        tqdm.pandas(desc=f"Processing Semantic-Only for {model_name} on {data_test}")
        test_data[f'SemanticOnly_{model_name}'] = test_data.index.to_series().progress_apply(
            lambda idx: semantic_only_generator.generate_semantic_only_text(
                test_data.loc[idx, 'target'],
                GENERATION_CONFIGS,
                target_stance
            )
        )
        
        # 应用其他方法
        # tqdm.pandas(desc=f"Processing OURS-P for {model_name} on {data_test}")
        # test_data[f'OURS-P_{model_name}'] = test_data.index.to_series().progress_apply(
        #     lambda idx: datg_text_generator.generate_with_prefix_prompt(
        #         test_data.loc[idx, 'prompt'], 
        #         important_nodes_positive_dict[idx], 
        #         important_nodes_negative_dict[idx], 
        #         GENERATION_CONFIGS
        #     )
        # )

        # tqdm.pandas(desc=f"Processing OURS-L for {model_name} on {data_test}")
        # test_data[f'OURS-L_{model_name}'] = test_data.index.to_series().progress_apply(
        #     lambda idx: datg_text_generator.generate_with_logits_processor(
        #         test_data.loc[idx, 'prompt'], 
        #         important_nodes_positive_dict[idx], 
        #         important_nodes_negative_dict[idx], 
        #         BOOST_VALUE, 
        #         AVOID_VALUE, 
        #         GENERATION_CONFIGS
        #     )
        # )

        # tqdm.pandas(desc=f"Processing CONTINUE for {model_name} on {data_test}")
        # test_data[f'CONTINUE_{model_name}'] = test_data['target'].progress_apply(
        #     lambda x: datg_text_generator.generate_texts(x, GENERATION_CONFIGS, num_return_sequences=1)
        # )

        # tqdm.pandas(desc=f"Processing INJECTION for {model_name} on {data_test}")
        # test_data[f'INJECTION_{model_name}'] = test_data['target'].progress_apply(
        #     lambda x: datg_text_generator.generate_texts(
        #         TASK_CONFIGURATIONS[task_name]["positive_prompt"] + x, 
        #         GENERATION_CONFIGS, 
        #         num_return_sequences=1
        #     )
        # )

        # tqdm.pandas(desc=f"Processing PREADD for {model_name} on {data_test}")
        # test_data[f'PREADD_{model_name}'] = test_data['target'].progress_apply(
        #     lambda x: preadd_text_generator.generate_preadd_text(
        #         x, 
        #         TASK_CONFIGURATIONS[task_name]['negative_prompt'], 
        #         GENERATION_CONFIGS, 
        #         strength=PREADD_STRENGTH
        #     )
        # )

        # tqdm.pandas(desc=f"Processing FUDGE for {model_name} on {data_test}")
        # test_data[f'FUDGE_{model_name}'] = test_data['target'].progress_apply(
        #     lambda x: fudge_text_generator.generate_fudge_texts(
        #         x, 
        #         GENERATION_CONFIGS, 
        #         fudge_alpha=FUDGE_ALPHA, 
        #         task=task_name
        #     )
        # )

        # 保存结果
        result_dir = task_config['result_dir']
        result_file_name = f"{result_dir}/semantic_only_{model_name}_{data_test}_results.json"

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        test_data.to_json(result_file_name, orient='records', lines=True, force_ascii=False)
        print(f"Results saved to {result_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model and task configurations.')
    parser.add_argument("--model_name", type=str, choices=list(MODEL_PATHS.keys()), help="Available model names.")
    parser.add_argument("--task_name", type=str, choices=["2Positive", "2Negative"], help="Available task names.")
    
    args = parser.parse_args()

    main(args.task_name, args.model_name)
    # main('2Positive', 'phi2_3B_Base')  # 如需快速测试可使用此行