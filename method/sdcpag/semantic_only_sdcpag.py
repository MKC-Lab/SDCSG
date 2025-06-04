import torch
from transformers import LogitsProcessorList, LogitsProcessor
from method.datg.datg import DatgTextGenerator  # Import DatgTextGenerator
import json
from method.sdcpag.sdcpag import SemanticAttributeGraphGenerator  # 复用原始的SemanticAttributeGraphGenerator

class SemanticOnlyProcessor:
    """
    仅使用语义级属性图的处理器（去掉词级指导信息）
    """
    
    def __init__(self, resource_manager):
        """初始化仅语义处理器"""
        # 只创建语义级图处理器
        self.semantic_processor = SemanticAttributeGraphGenerator(resource_manager)
    
    def process_input(self, input_text, target_stance, initial_texts=None):
        """
        处理输入文本，只生成语义级指导信息
        
        Args:
            input_text: 输入文本
            target_stance: 目标立场
            initial_texts: 忽略初始文本
            
        Returns:
            tuple: (空词级关键词, 语义图)
        """
        # 只生成语义级属性图
        semantic_graph = self.semantic_processor.generate_semantic_graph(input_text, target_stance)
        
        # 返回空的词级指导和语义图
        return ([], []), semantic_graph
    
    def create_compositional_prompt(self, input_text, word_level_guidance, semantic_graph, target_stance):
        """
        创建组合提示，仅使用语义级信息
        
        Args:
            input_text: 输入文本
            word_level_guidance: 不使用
            semantic_graph: 语义级属性图
            target_stance: 目标立场
            
        Returns:
            str: 组合提示
        """
        # 提取语义级信息
        entities = semantic_graph.get("entities", [])[:3]
        entities_str = ", ".join(entities) if entities else ""
        
        attributes = semantic_graph.get("attributes", [])[:3]
        attributes_str = ", ".join(attributes) if attributes else ""
        
        # 提取关系信息
        relationships = []
        if "relationships" in semantic_graph and semantic_graph["relationships"]:
            rel_dict = semantic_graph["relationships"][0]
            for entity, relations in rel_dict.items():
                for relation, targets in relations.items():
                    for target in targets:
                        relationships.append(f"{entity} {relation} {target}")
        
        relations_str = "; ".join(relationships[:2]) if relationships else ""
        
        # 根据立场定制提示
        stance_description = "positive and supportive" if target_stance == "positive" else "strongly opposed and critical"
        
        # 创建组合提示 (不包含词级关键词)
        prompt = f"Task: Write a {stance_description} response.\n\n"
        
        if entities_str:
            prompt += f"Key entities: {entities_str}\n"
        
        if attributes_str:
            prompt += f"Key attributes: {attributes_str}\n"
        
        if relations_str:
            prompt += f"Key relationships: {relations_str}\n"
        
        prompt += f"\nInput: {input_text}\n\nResponse:"
        
        return prompt

class SemanticOnlyTextGenerator(DatgTextGenerator):
    """
    基于仅语义级属性图的文本生成器
    """
    
    def __init__(self, resource_manager, num_sentences=15, nodes_num=10):
        """初始化仅语义文本生成器"""
        super().__init__(resource_manager, num_sentences, nodes_num)
        self.sdcpag_processor = SemanticOnlyProcessor(resource_manager)
        
        # 确保tokenizer设置正确
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            print("Fixing tokenizer configuration...")
            self.tokenizer.pad_token = "[PAD]"
            # 如果模型词汇表中没有[PAD]，则扩展词汇表
            if "[PAD]" not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                # 如果模型词汇表大小变化，需要调整模型嵌入
                self.model.resize_token_embeddings(len(self.tokenizer))
    
    def generate_semantic_only_text(self, prompt, generation_configs, target_stance="positive"):
        """
        仅使用语义级属性图生成文本
        
        Args:
            prompt: 输入提示
            generation_configs: 生成配置
            target_stance: 目标立场
            
        Returns:
            str: 生成的文本
        """
        try:
            # 确保输入不为空
            if not prompt or prompt.strip() == "":
                print(f"Warning: Empty prompt received")
                return "Empty input provided"
                
            # 生成语义级指导信息
            word_guidance, semantic_graph = self.sdcpag_processor.process_input(
                prompt, target_stance, None)
            
            # 检查语义图是否为空
            if not semantic_graph or (not semantic_graph.get("entities") and not semantic_graph.get("attributes")):
                print(f"Warning: Empty semantic graph for prompt: {prompt[:50]}...")
                # 创建基本语义图以确保有内容可用
                semantic_graph = {
                    "entities": [word for word in prompt.split()[:3] if len(word) > 3],
                    "attributes": ["important", "relevant", "key"],
                    "relationships": [{"topic": {"related_to": ["discussion"]}}]
                }
            
            # 创建增强提示
            enhanced_prompt = self.sdcpag_processor.create_compositional_prompt(
                prompt, word_guidance, semantic_graph, target_stance)
            
            print(f"Enhanced prompt: {enhanced_prompt[:100]}...")
            
            inputs = self.tokenizer(enhanced_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=generation_configs.get('max_new_tokens', 100),
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 去除原始提示
                if result.startswith(enhanced_prompt):
                    result = result[len(enhanced_prompt):].strip()
                
                # 如果结果为空，使用备选方案
                if not result or not result.strip():
                    print("Warning: Empty primary generation result. Using fallback...")
                    fallback_prompt = f"Write a {'positive' if target_stance == 'positive' else 'negative'} response to: {prompt}"
                    
                    fallback_inputs = self.tokenizer(fallback_prompt, return_tensors="pt")
                    fallback_input_ids = fallback_inputs["input_ids"].to(self.model.device)
                    fallback_attention_mask = fallback_inputs["attention_mask"].to(self.model.device)
                    
                    fallback_outputs = self.model.generate(
                        input_ids=fallback_input_ids,
                        attention_mask=fallback_attention_mask,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=1
                    )
                    
                    result = self.tokenizer.decode(fallback_outputs[0], skip_special_tokens=True)
                    
                    # 只保留生成的部分
                    if result.startswith(fallback_prompt):
                        result = result[len(fallback_prompt):].strip()
                    
                    if not result or not result.strip():
                        return f"I {'support' if target_stance == 'positive' else 'oppose'} this statement."
                
                return result
                
            except Exception as e:
                print(f"Error in primary generation: {e}")
                return f"Response to: {prompt[:30]}..."
                
        except Exception as e:
            print(f"Error in generate_semantic_only_text: {e}")
            # 返回一个基本响应以防止空输出
            return f"Response to: {prompt[:20]}..."