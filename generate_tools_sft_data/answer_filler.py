# 调用工具，填充回答案
import json
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import sys

# 导入工具函数
sys.path.append('/home/ubuntu/50T/multi_mcp_server')  # 根据实际路径调整
from sci_mcp import (
    search_advanced_pubchem,
    predict_reaction_outcome_rxn,
    predict_reaction_topn_rxn,
    predict_reaction_properties_rxn,
    predict_retrosynthesis_rxn,
    extract_reaction_actions_rxn,
    get_all_tools
)

# 导入RDKit工具 - 由于有多个模块，我们需要动态导入
from sci_mcp.chemistry_mcp.rdkit_tools import rdkit_basic, rdkit_advanced, rdkit_estate, rdkit_extended


class ToolCallProcessor:
    """处理工具调用的类"""
    
    def __init__(self):
        """初始化工具映射"""
        self.tools = {}
        self._load_tools()
    
    def _load_tools(self):
        """加载所有可用的工具"""
        # PubChem工具
        self.tools['search_advanced_pubchem'] = search_advanced_pubchem
        
        # RXN工具
        self.tools['predict_reaction_outcome_rxn'] = predict_reaction_outcome_rxn
        self.tools['predict_reaction_topn_rxn'] = predict_reaction_topn_rxn
        self.tools['predict_reaction_properties_rxn'] = predict_reaction_properties_rxn
        self.tools['predict_retrosynthesis_rxn'] = predict_retrosynthesis_rxn
        self.tools['extract_reaction_actions_rxn'] = extract_reaction_actions_rxn
        
        # RDKit基础工具
        for attr_name in dir(rdkit_basic):
            attr = getattr(rdkit_basic, attr_name)
            if callable(attr) and hasattr(attr, 'is_llm_tool'):
                self.tools[attr_name] = attr
        
        # RDKit高级工具
        for attr_name in dir(rdkit_advanced):
            attr = getattr(rdkit_advanced, attr_name)
            if callable(attr) and hasattr(attr, 'is_llm_tool'):
                self.tools[attr_name] = attr
        
        # RDKit EState工具
        for attr_name in dir(rdkit_estate):
            attr = getattr(rdkit_estate, attr_name)
            if callable(attr) and hasattr(attr, 'is_llm_tool'):
                self.tools[attr_name] = attr
        
        # RDKit扩展工具
        for attr_name in dir(rdkit_extended):
            attr = getattr(rdkit_extended, attr_name)
            if callable(attr) and hasattr(attr, 'is_llm_tool'):
                self.tools[attr_name] = attr
        
        print(f"已加载 {len(self.tools)} 个工具函数")
    
    def extract_tool_call(self, answer: str) -> Dict[str, Any]:
        """
        从answer中提取tool_call信息
        
        Args:
            answer: 包含<tool_call>标签的字符串
            
        Returns:
            包含name和arguments的字典，如果没有找到则返回None
        """
        # 使用正则表达式提取<tool_call>标签中的内容
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, answer, re.DOTALL)
        
        if not match:
            return None
        
        # 提取JSON字符串
        json_str = match.group(1).strip()
        
        try:
            # 解析JSON
            tool_call_data = json.loads(json_str)
            
            # 验证必需字段
            if 'name' not in tool_call_data or 'arguments' not in tool_call_data:
                print(f"警告: tool_call缺少必需字段: {tool_call_data}")
                return None
            
            # arguments可能是字符串形式的JSON，需要再次解析
            if isinstance(tool_call_data['arguments'], str):
                tool_call_data['arguments'] = json.loads(tool_call_data['arguments'])
            
            return tool_call_data
        
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始字符串: {json_str}")
            return None
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        调用指定的工具函数
        
        Args:
            name: 工具函数名称
            arguments: 工具函数参数
            
        Returns:
            工具函数的执行结果
        """
        if name not in self.tools:
            return f"错误: 未找到工具函数 '{name}'"
        
        tool_func = self.tools[name]
        
        try:
            # 检查函数是否是异步的
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**arguments)
            else:
                result = tool_func(**arguments)
            
            return result
        
        except Exception as e:
            return f"工具调用错误: {str(e)}"
    
    def update_answer_with_result(self, answer: str, result: str) -> str:
        """
        将工具调用结果添加到answer中
        
        Args:
            answer: 原始answer字符串
            result: 工具调用结果
            
        Returns:
            更新后的answer字符串
        """
        # 在</tool_call>标签后添加结果
        pattern = r'(</tool_call>)'
        replacement = r'\1\n\n### 工具调用结果\n\n' + result
        
        updated_answer = re.sub(pattern, replacement, answer, count=1)
        return updated_answer
    
    async def process_json_file(self, input_file: str, output_file: str = None):
        """
        处理JSON文件
        
        Args:
            input_file: 输入JSON文件路径
            output_file: 输出JSON文件路径，如果为None则覆盖输入文件
        """
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 判断是单个对象还是对象列表
        if isinstance(data, dict):
            data_list = [data]
        elif isinstance(data, list):
            data_list = data
        else:
            print("错误: JSON文件格式不正确")
            return
        
        # 处理每个数据项
        for i, item in enumerate(data_list):
            print(f"\n处理第 {i+1}/{len(data_list)} 项...")
            
            if 'answer' not in item:
                print(f"警告: 第 {i+1} 项缺少 'answer' 字段")
                continue
            
            answer = item['answer']
            
            # 提取tool_call
            tool_call = self.extract_tool_call(answer)
            
            if not tool_call:
                print(f"第 {i+1} 项未找到有效的tool_call")
                continue
            
            print(f"工具名称: {tool_call['name']}")
            print(f"工具参数: {tool_call['arguments']}")
            
            # 调用工具
            result = await self.call_tool(tool_call['name'], tool_call['arguments'])
            
            print(f"工具调用完成，结果长度: {len(result)} 字符")
            
            # 更新answer
            item['answer'] = self.update_answer_with_result(answer, result)
            
            # 可选：将结果也存储在单独的字段中
            item['tool_result'] = result
        
        # 保存结果
        output_path = output_file if output_file else input_file
        
        # 如果原始数据是单个对象，保存为单个对象
        save_data = data_list[0] if isinstance(data, dict) else data_list
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成！结果已保存到: {output_path}")
    
    async def process_multiple_files(self, input_files: List[str], output_dir: str = None):
        """
        批量处理多个JSON文件
        
        Args:
            input_files: 输入JSON文件路径列表
            output_dir: 输出目录，如果为None则覆盖原文件
        """
        for input_file in input_files:
            print(f"\n{'='*60}")
            print(f"处理文件: {input_file}")
            print(f"{'='*60}")
            
            if output_dir:
                output_file = Path(output_dir) / Path(input_file).name
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = None
            
            await self.process_json_file(input_file, str(output_file) if output_file else None)


async def main():
    """主函数"""
    # 创建处理器
    processor = ToolCallProcessor()
    
    input_file = "/home/ubuntu/50T/fsy/json/generate_stf_data/response.json"  # 替换为实际文件路径
    output_file = "response.json"  # 可选，如果不指定则覆盖输入文件
    
    await processor.process_json_file(input_file, output_file)


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())