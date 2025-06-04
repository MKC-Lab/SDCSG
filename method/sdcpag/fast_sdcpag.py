import torch
import json
import re
from method.datg.datg import DatgTextGenerator
from method.sdcpag.sdcpag import SDCPAGProcessor

class FastSDCPAGTextGenerator(DatgTextGenerator):
    """
    优化版SD-CPAG文本生成器，专注于速度和效率
    """
    
    def __init__(self, resource_manager, num_sentences=5, nodes_num=5):
        """初始化优化版SD-CPAG文本生成器，使用更小的参数值"""
        super().__init__(resource_manager, num_sentences, nodes_num)
        self.model = resource_manager.model
        self.tokenizer = resource_manager.tokenizer
    
    def generate_texts_with_attention_mask(self, prompt, generation_configs, num_return_sequences=1):
        """带有明确attention_mask的文本生成方法"""
        generation_configs_updated = generation_configs.copy()
        generation_configs_updated['num_return_sequences'] = num_return_sequences
        
        # 编码输入并明确设置attention_mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # 设置pad_token_id以避免警告
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        outputs = self.model.generate(
            input_ids, 
            attention_mask=attention_mask,
            **generation_configs_updated
        )
        
        if num_return_sequences == 1:
            return self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        else:
            return [self.tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=True) for output in outputs]
    
    def generate_fast_semantic_graph(self, input_text, target_stance):
        """快速生成简化的语义图"""
        # 使用简化的提示，减少token数量
        stance_desc = "positive" if target_stance == "positive" else "negative"
        prompt = f"""Analyze this text and create a minimal attribute graph. Focus on {stance_desc} stance.
Text: "{input_text}"
JSON format with entities, attributes and key relationships only:"""

        # 使用更高效的生成参数
        fast_gen_config = {
            'max_new_tokens': 300,
            'temperature': 0.2,
            'do_sample': False,  # 使用贪婪解码
            'num_beams': 1       # 不使用波束搜索
        }
        
        # 生成语义图
        response = self.generate_texts_with_attention_mask(prompt, fast_gen_config)
        
        # 提取JSON部分
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # 简单的回退策略
                return self._create_basic_graph(input_text, target_stance)
        except:
            return self._create_basic_graph(input_text, target_stance)
    
    def _create_basic_graph(self, input_text, target_stance):
        """创建基本属性图"""
        # 提取可能的实体和属性（简单实现）
        words = re.findall(r'\b\w+\b', input_text)
        entities = [w for w in words if len(w) > 3][:5]
        
        return {
            "entities": entities,
            "attributes": ["positive" if target_stance == "positive" else "negative"],
            "relationships": [{}]
        }
    
    def generate_fast_sdcpag_text(self, prompt, generation_configs, target_stance="positive"):
        """
        快速版SD-CPAG文本生成方法
        
        Args:
            prompt: 输入提示
            generation_configs: 生成配置
            target_stance: 目标立场
            
        Returns:
            str: 生成的文本
        """
        # 1. 生成少量初始文本，节省时间
        fast_gen_config = generation_configs.copy()
        fast_gen_config['max_new_tokens'] = min(fast_gen_config.get('max_new_tokens', 128), 128)
        
        initial_texts = self.generate_texts_with_attention_mask(
            prompt, fast_gen_config, self.num_sentences)
        
        # 2. 提取重要节点（词级分析）
        positive_graph, negative_graph = self.graph_processor.process_sentences_dual_graph(
            initial_texts, 
            task="2Positive" if target_stance == "positive" else "2Negative"
        )
        
        if target_stance == "positive":
            boost_keywords = self.graph_processor.find_important_nodes(positive_graph, self.nodes_num)
            avoid_keywords = self.graph_processor.find_important_nodes(negative_graph, self.nodes_num)
        else:
            boost_keywords = self.graph_processor.find_important_nodes(negative_graph, self.nodes_num)
            avoid_keywords = self.graph_processor.find_important_nodes(positive_graph, self.nodes_num)
        
        # 3. 快速生成语义图（简化版）
        semantic_graph = self.generate_fast_semantic_graph(prompt, target_stance)
        
        # 4. 创建组合提示
        enhanced_prompt = self._create_fast_compositional_prompt(
            prompt, boost_keywords, avoid_keywords, semantic_graph, target_stance)
        
        # 5. 生成最终文本
        result = self.generate_texts_with_attention_mask(enhanced_prompt, generation_configs)
        return result
    
    def _create_fast_compositional_prompt(self, prompt, boost_keywords, avoid_keywords, semantic_graph, target_stance):
        """创建简化版组合提示"""
        # 格式化词级关键词
        boost_str = ", ".join(boost_keywords[:3]) if boost_keywords else ""
        avoid_str = ", ".join(avoid_keywords[:3]) if avoid_keywords else ""
        
        # 提取语义图中的关键信息
        entities = semantic_graph.get("entities", [])[:3]
        entities_str = ", ".join(entities) if entities else ""
        
        attributes = semantic_graph.get("attributes", [])[:2]
        attributes_str = ", ".join(attributes) if attributes else ""
        
        # 构建精简提示
        stance_desc = "positive and supportive" if target_stance == "positive" else "negative and critical"
        
        enhanced_prompt = f"Task: Write a {stance_desc} response.\n"
        
        if boost_str:
            enhanced_prompt += f"Emphasize: {boost_str}\n"
        
        if entities_str:
            enhanced_prompt += f"Key entities: {entities_str}\n"
        
        if attributes_str:
            enhanced_prompt += f"Key attributes: {attributes_str}\n"
        
        if avoid_str:
            enhanced_prompt += f"Avoid: {avoid_str}\n"
        
        enhanced_prompt += f"\nInput: {prompt}\n\nResponse:"
        
        return enhanced_prompt