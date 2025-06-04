import torch
from transformers import LogitsProcessorList, LogitsProcessor
from method.datg.datg import DatgTextGenerator  # Import DatgTextGenerator
import json

class SemanticAttributeGraphGenerator:
    """
    通过思维链方法生成语义级属性控制图的处理器
    """
    
    def __init__(self, resource_manager):
        """初始化语义图生成器"""
        self.model = resource_manager.model
        self.tokenizer = resource_manager.tokenizer
    
    def generate_semantic_graph(self, input_text, target_stance):
        """
        使用思维链方法生成语义级属性图
        
        Args:
            input_text: 输入文本
            target_stance: 目标立场 ("positive" 或 "negative")
            
        Returns:
            dict: 属性图数据结构
        """
        # 构建思维链提示，引导模型分析文本并创建属性图
        stance_desc = "positive" if target_stance == "positive" else "negative"
        if target_stance == "negative":
            prompt = f"""For the provided text, generate a attribute graph in JSON format that helps me create a STRONG OPPOSING response.
        Focus on identifying:
        1. Questionable claims or potential flaws in the text
        2. Alternative perspectives that challenge the original view
        3. Strong emotional words that express disagreement or criticism
        
        Input Text: "{input_text}"
        Task: Generate content with a strongly negative/opposing stance.
        
        Think step by step:
        1. What are the main claims or assumptions in this text that can be challenged?
        2. What counterarguments or opposite viewpoints exist?
        3. What rhetorical devices could be used to express strong disagreement?
        
        Output your analysis in the following JSON format:
        {{
          "entities": ["entity1", "entity2", ...],
          "attributes": ["criticism1", "objection2", ...],
          "relationships": [{{
            "entity1": {{"criticism": ["entity2"]}},
            "claim": {{"rebuttal": ["counterpoint1", "counterpoint2"]}}
          }}]
        }}
        """
        else:
            prompt = f"""For the provided text and its associated task, generate a attribute graph in JSON format that includes the following:
1. Entities that are relevant to understanding the user's topic.
2. Words that reveal user's sentiment tendencies.
3. Relationships between entities that help to clarify the context and the stance of the user.

Input Text: "{input_text}"
Task: Generate content with a {stance_desc} stance.

Think step by step:
1. First, identify all relevant entities in the text.
2. Next, determine key attributes or sentiment words.
3. Finally, analyze how these entities relate to each other.

Output your analysis in the following JSON format:
{{
  "entities": ["entity1", "entity2", ...],
  "attributes": ["attribute1", "attribute2", ...],
  "relationships": [{{
    "entity1": {{"relation": ["entity2"]}},
    "entity3": {{"relation": ["entity4", "entity5"]}}
  }}]
}}
"""

        # 使用模型生成思维链分析和属性图
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, 
                max_length=2048, 
                temperature=0.1,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 提取JSON部分
        try:
            # 尝试找到JSON格式部分
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return self._create_fallback_graph(input_text)
        except:
            return self._create_fallback_graph(input_text)
    
    def _create_fallback_graph(self, input_text):
        """创建基本的属性图结构"""
        return {
            "entities": [word for word in input_text.split()[:3] if len(word) > 3],
            "attributes": ["neutral"],
            "relationships": [{}]
        }

class SDCPAGProcessor:
    """
    SD-CPAG方法的处理器，结合词级和语义级属性图
    """
    
    def __init__(self, resource_manager):
        """初始化SD-CPAG处理器"""
        # 复用DATG的词级图处理器
        from method.datg.datg import GraphProcessor
        self.word_processor = GraphProcessor(resource_manager)
        # 创建语义级图处理器
        self.semantic_processor = SemanticAttributeGraphGenerator(resource_manager)
    
    def process_input(self, input_text, target_stance, initial_texts=None):
        """
        处理输入文本，生成词级和语义级指导信息
        
        Args:
            input_text: 输入文本
            target_stance: 目标立场
            initial_texts: 可选的初始生成文本，用于构建词级图
            
        Returns:
            tuple: (词级关键词, 语义图)
        """
        # 1. 生成语义级属性图
        semantic_graph = self.semantic_processor.generate_semantic_graph(input_text, target_stance)
        
        # 2. 如果有初始文本，构建词级图并提取关键词
        if initial_texts:
            positive_graph, negative_graph = self.word_processor.process_sentences_dual_graph(
                initial_texts, 
                task="2Positive" if target_stance == "positive" else "2Negative"
            )
            
            if target_stance == "positive":
                boost_keywords = self.word_processor.find_important_nodes(positive_graph, 10)
                avoid_keywords = self.word_processor.find_important_nodes(negative_graph, 10)
            else:
                boost_keywords = self.word_processor.find_important_nodes(negative_graph, 10)
                avoid_keywords = self.word_processor.find_important_nodes(positive_graph, 10)
                
            word_level_guidance = (boost_keywords, avoid_keywords)
        else:
            word_level_guidance = ([], [])
        
        return word_level_guidance, semantic_graph
    
    def create_compositional_prompt(self, input_text, word_level_guidance, semantic_graph, target_stance):
        """
        创建组合提示，结合词级和语义级信息
        
        Args:
            input_text: 输入文本
            word_level_guidance: 词级关键词 (boost_keywords, avoid_keywords)
            semantic_graph: 语义级属性图
            target_stance: 目标立场
            
        Returns:
            str: 组合提示
        """
        boost_keywords, avoid_keywords = word_level_guidance
        
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
        
        # 组合词级关键词
        boost_str = ", ".join(boost_keywords[:5]) if boost_keywords else ""
        avoid_str = ", ".join(avoid_keywords[:5]) if avoid_keywords else ""
        
        # 根据立场定制提示
        stance_description = "positive and supportive" if target_stance == "positive" else "strongly opposed and critical"
        
        # 创建组合提示
        prompt = f"Task: Write a {stance_description} response.\n\n"
        
        if boost_str:
            prompt += f"Emphasize concepts: {boost_str}\n"
        
        if entities_str:
            prompt += f"Key entities: {entities_str}\n"
        
        if attributes_str:
            prompt += f"Key attributes: {attributes_str}\n"
        
        if relations_str:
            prompt += f"Key relationships: {relations_str}\n"
        
        if avoid_str:
            prompt += f"Avoid discussing: {avoid_str}\n"
        
        prompt += f"\nInput: {input_text}\n\nResponse:"
        
        return prompt

class SDCPAGTextGenerator(DatgTextGenerator):
    """
    基于SD-CPAG方法的文本生成器，扩展DATG文本生成器
    """
    
    def __init__(self, resource_manager, num_sentences=15, nodes_num=10):
        """初始化SD-CPAG文本生成器"""
        super().__init__(resource_manager, num_sentences, nodes_num)
        self.sdcpag_processor = SDCPAGProcessor(resource_manager)
    
    def generate_sdcpag_text(self, prompt, generation_configs, target_stance="positive"):
        """
        使用SD-CPAG方法生成文本
        
        Args:
            prompt: 输入提示
            generation_configs: 生成配置
            target_stance: 目标立场
            
        Returns:
            str: 生成的文本
        """
        # 1. 生成初始文本
        initial_texts = self.generate_texts(prompt, generation_configs, self.num_sentences)
        
        # 2. 处理输入，获取词级和语义级指导信息
        word_guidance, semantic_graph = self.sdcpag_processor.process_input(
            prompt, target_stance, initial_texts)
        
        # 3. 创建组合提示
        enhanced_prompt = self.sdcpag_processor.create_compositional_prompt(
            prompt, word_guidance, semantic_graph, target_stance)
        
        # 4. 使用logits处理器生成文本
        boost_keywords, avoid_keywords = word_guidance
        return self.generate_with_logits_processor(
            enhanced_prompt, boost_keywords, avoid_keywords, 3.0, 4.0, generation_configs)