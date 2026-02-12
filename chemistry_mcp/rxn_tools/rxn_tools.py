"""
RXN Tools Module

This module provides tools for chemical reaction prediction and analysis
using the IBM RXN for Chemistry API through the rxn4chemistry package.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Union, Optional, Any, Tuple

from rxn4chemistry import RXN4ChemistryWrapper
from ...core.llm_tools import llm_tool
from ...core.config import Chemistry_Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from rich.console import Console
console= Console()
# Constants
DEFAULT_MAX_RESULTS = 3
DEFAULT_TIMEOUT = 180  # seconds - increased from 60 to 180


def _get_rxn_wrapper() -> RXN4ChemistryWrapper:
    """
    Get an initialized RXN4Chemistry wrapper with API key and project set.

    Returns:
        Initialized RXN4ChemistryWrapper instance with project set

    Raises:
        ValueError: If API key is not available
    """
    # Try to get API key from environment or config
    api_key = Chemistry_Config.RXN4_CHEMISTRY_KEY or os.environ.get("RXN_API_KEY")

    if not api_key:
        raise ValueError("Error: RXN API key not found. Please set the RXN_API_KEY environment variable.")

    # Initialize the wrapper
    wrapper = RXN4ChemistryWrapper(api_key=api_key)

    try:
        # Create a new project
        project_name = f"RXN_Tools_Project_{os.getpid()}"  # Add process ID to make name unique
        project_response = wrapper.create_project(project_name)

        # Extract project ID from response
        # The API response format is nested: {'response': {'payload': {'id': '...'}}
        if project_response and isinstance(project_response, dict):
            # Try to extract project ID from different possible response formats
            project_id = None

            # Direct format: {"project_id": "..."}
            if "project_id" in project_response:
                project_id = project_response["project_id"]

            # Nested format: {"response": {"payload": {"id": "..."}}}
            elif "response" in project_response and isinstance(project_response["response"], dict):
                payload = project_response["response"].get("payload", {})
                if isinstance(payload, dict) and "id" in payload:
                    project_id = payload["id"]

            if project_id:
                wrapper.set_project(project_id)
                logger.info(f"RXN project '{project_name}' created and set successfully with ID: {project_id}")
            else:
                logger.warning(f"Could not extract project ID from response: {project_response}")
        else:
            logger.warning(f"Unexpected project creation response: {project_response}")
    except Exception as e:
        logger.error(f"Error creating RXN project: {e}")

    # Check if project is set
    if not hasattr(wrapper, "project_id") or not wrapper.project_id:
        logger.warning("No project set. Some API calls may fail.")

    return wrapper



def _format_reaction_markdown(reactants: str, products: List[str],
                             confidence: Optional[List[float]] = None) -> str:
    """
    Format reaction results as Markdown.

    Args:
        reactants: SMILES of reactants
        products: List of product SMILES
        confidence: Optional list of confidence scores

    Returns:
        Formatted Markdown string
    """
    markdown = f"## 反应预测结果\n\n"
    markdown += f"### 输入反应物\n\n`{reactants}`\n\n"

    markdown += f"### 预测产物\n\n"

    for i, product in enumerate(products):
        conf_str = f" (置信度: {confidence[i]:.2f})" if confidence and i < len(confidence) else ""
        markdown += f"{i+1}. `{product}`{conf_str}\n"

    return markdown


@llm_tool(name="predict_reaction_outcome_rxn",
          description="Predict chemical reaction outcomes for given reactants using IBM RXN for Chemistry API")
async def predict_reaction_outcome_rxn(reactants: str) -> str:
    """
    Predict chemical reaction outcomes for given reactants.

    This function uses the IBM RXN for Chemistry API to predict the most likely
    products formed when the given reactants are combined.

    Args:
        reactants: SMILES notation of reactants, multiple reactants separated by dots (.).

    Returns:
        Formatted Markdown string containing the predicted reaction results.

    Examples:
        >>> predict_reaction_outcome_rxn("BrBr.c1ccc2cc3ccccc3cc2c1")
        # Returns predicted products of bromine and anthracene reaction
    """
    try:
        # Get RXN wrapper
        wrapper = _get_rxn_wrapper()

        # Clean input
        reactants = reactants.strip()

        # Submit prediction
        response = await asyncio.to_thread(
            wrapper.predict_reaction, reactants
        )

        if not response or "prediction_id" not in response:
            return "Error: 无法提交反应预测请求"

        # 直接获取结果，而不是通过_wait_for_result
        results = await asyncio.to_thread(
            wrapper.get_predict_reaction_results,
            response["prediction_id"]
        )

        # Extract products
        try:
            attempts = results.get("response", {}).get("payload", {}).get("attempts", [])
            if not attempts:
                return "Error: 未找到预测结果"

            # Get the top predicted product
            product_smiles = attempts[0].get("smiles", "")
            confidence = attempts[0].get("confidence", None)

            # Format results
            return _format_reaction_markdown(
                reactants,
                [product_smiles] if product_smiles else ["无法预测产物"],
                [confidence] if confidence is not None else None
            )

        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing prediction results: {e}")
            return f"Error: 解析预测结果时出错: {str(e)}"

    except Exception as e:
        logger.error(f"Error in predict_reaction_outcome: {e}")
        return f"Error: {str(e)}"


@llm_tool(name="predict_reaction_topn_rxn",
          description="Predict multiple possible products for chemical reactions using IBM RXN for Chemistry API")
async def predict_reaction_topn_rxn(reactants: Union[str, List[str], List[List[str]]], topn: int = 3) -> str:
    """
    Predict multiple possible products for chemical reactions.

    This function uses the IBM RXN for Chemistry API to predict multiple products
    that may be formed from given reactants, ranked by likelihood. Suitable for
    scenarios where multiple reaction pathways need to be considered.

    Args:
        reactants: Reactants in one of the following formats:
            - String: SMILES notation for a single reaction, multiple reactants separated by dots (.)
            - List of strings: Multiple reactants for a single reaction, each reactant as a SMILES string
            - List of lists of strings: Multiple reactions, each reaction composed of multiple reactant SMILES strings
        topn: Number of predicted products to return for each reaction, default is 3.

    Returns:
        Formatted Markdown string containing multiple predicted reaction results.

    Examples:
        >>> predict_reaction_topn_rxn("BrBr.c1ccc2cc3ccccc3cc2c1", 5)
        # Returns top 5 possible products for bromine and anthracene reaction

        >>> predict_reaction_topn_rxn(["BrBr", "c1ccc2cc3ccccc3cc2c1"], 3)
        # Returns top 3 possible products for bromine and anthracene reaction

        >>> predict_reaction_topn_rxn([
        ...     ["BrBr", "c1ccc2cc3ccccc3cc2c1"],
        ...     ["BrBr", "c1ccc2cc3ccccc3cc2c1CCO"]
        ... ], 3)
        # Returns top 3 possible products for two different reactions
    """
    try:
        # Get RXN wrapper
        wrapper = _get_rxn_wrapper()

        # Validate topn
        if topn < 1:
            topn = 1
        elif topn > 10:
            topn = 10
            logger.warning("topn限制为最大10个结果")

        # Process input to create precursors_lists
        precursors_lists = []

        if isinstance(reactants, str):
            # Single reaction as string (e.g., "BrBr.c1ccc2cc3ccccc3cc2c1")
            reactants = reactants.strip()
            precursors_lists = [reactants.split(".")]
            # For display in results
            reactants_display = [reactants]

        elif isinstance(reactants, list):
            if all(isinstance(r, str) for r in reactants):
                # Single reaction as list of strings (e.g., ["BrBr", "c1ccc2cc3ccccc3cc2c1"])
                precursors_lists = [reactants]
                # For display in results
                reactants_display = [".".join(reactants)]

            elif all(isinstance(r, list) for r in reactants):
                # Multiple reactions as list of lists (e.g., [["BrBr", "c1ccc2cc3ccccc3cc2c1"], ["BrBr", "c1ccc2cc3ccccc3cc2c1CCO"]])
                precursors_lists = reactants
                # For display in results
                reactants_display = [".".join(r) for r in reactants]

            else:
                return "Error: 反应物列表格式无效，必须是字符串列表或字符串列表的列表"
        else:
            return "Error: 反应物参数类型无效，必须是字符串或列表"

        # Submit prediction
        response = await asyncio.to_thread(
            wrapper.predict_reaction_batch_topn,
            precursors_lists=precursors_lists,
            topn=topn
        )

        if not response or "task_id" not in response:
            return "Error: 无法提交多产物反应预测请求"

        # 直接获取结果，不使用循环等待
        results = await asyncio.to_thread(
            wrapper.get_predict_reaction_batch_topn_results,
            response["task_id"]
        )

        # Extract products
        try:
            # 记录结果的结构，以便调试
            logger.info(f"Results structure: {results.keys()}")

            # 更灵活地获取结果，使用get方法并提供默认值
            reaction_results = results.get("result", [])

            # 如果结果为空，尝试其他可能的键
            if not reaction_results and "predictions" in results:
                reaction_results = results.get("predictions", [])
                logger.info("Using 'predictions' key instead of 'result'")

            # 如果结果仍然为空，尝试直接使用整个结果
            if not reaction_results and isinstance(results, list):
                reaction_results = results
                logger.info("Using entire results as list")

            if not reaction_results:
                logger.warning(f"No reaction results found. Available keys: {results.keys()}")
                return "Error: 未找到预测结果。请检查API响应格式。"

            # Format results for all reactions
            markdown = "## 反应预测结果\n\n"

            # 确保reaction_results和reactants_display长度匹配
            if len(reaction_results) != len(reactants_display):
                logger.warning(f"Mismatch between results ({len(reaction_results)}) and reactants ({len(reactants_display)})")
                # 如果不匹配，使用较短的长度
                min_len = min(len(reaction_results), len(reactants_display))
                reaction_results = reaction_results[:min_len]
                reactants_display = reactants_display[:min_len]

            for i, (reaction_result, reactants_str) in enumerate(zip(reaction_results, reactants_display), 1):
                if not reaction_result:
                    markdown += f"### 反应 {i}: 未找到预测结果\n\n"
                    continue

                # 记录每个反应结果的结构
                logger.info(f"Reaction {i} result structure: {type(reaction_result)}")

                # Extract products and confidences for this reaction
                products = []
                confidences = []

                # 处理不同格式的反应结果
                if isinstance(reaction_result, list):
                    # 标准格式：列表中的每个项目是一个预测
                    for pred in reaction_result:
                        if isinstance(pred, dict) and "smiles" in pred:
                            # 检查smiles是否为列表
                            if isinstance(pred["smiles"], list) and pred["smiles"]:
                                products.append(pred["smiles"][0])  # 取列表中的第一个元素
                            else:
                                products.append(pred["smiles"])
                            confidences.append(pred.get("confidence", 0.0))
                elif isinstance(reaction_result, dict):
                    # 根据用户反馈，检查是否有'results'键
                    if "results" in reaction_result:
                        # 遍历results列表
                        for pred in reaction_result.get("results", []):
                            if isinstance(pred, dict) and "smiles" in pred:
                                # 检查smiles是否为列表
                                if isinstance(pred["smiles"], list) and pred["smiles"]:
                                    products.append(pred["smiles"][0])  # 取列表中的第一个元素
                                else:
                                    products.append(pred["smiles"])
                                confidences.append(pred.get("confidence", 0.0))
                    # 替代格式：字典中直接包含预测
                    elif "smiles" in reaction_result:
                        # 检查smiles是否为列表
                        if isinstance(reaction_result["smiles"], list) and reaction_result["smiles"]:
                            products.append(reaction_result["smiles"][0])  # 取列表中的第一个元素
                        else:
                            products.append(reaction_result["smiles"])
                        confidences.append(reaction_result.get("confidence", 0.0))
                    # 另一种可能的格式
                    elif "products" in reaction_result:
                        for prod in reaction_result.get("products", []):
                            if isinstance(prod, dict) and "smiles" in prod:
                                # 检查smiles是否为列表
                                if isinstance(prod["smiles"], list) and prod["smiles"]:
                                    products.append(prod["smiles"][0])  # 取列表中的第一个元素
                                else:
                                    products.append(prod["smiles"])
                                confidences.append(prod.get("confidence", 0.0))

                # Add results for this reaction
                markdown += f"### 反应 {i}\n\n"
                markdown += f"**输入反应物:** `{reactants_str}`\n\n"

                if products:
                    markdown += "**预测产物:**\n\n"
                    for j, (product, confidence) in enumerate(zip(products, confidences), 1):
                        markdown += f"{j}. `{product}` (置信度: {confidence:.2f})\n"
                else:
                    markdown += "**预测产物:** 无法解析产物结构\n\n"
                    # 添加原始结果以便调试
                    markdown += f"**原始结果:** `{reaction_result}`\n\n"

                markdown += "\n"

            return markdown

        except Exception as e:
            logger.error(f"Error parsing topn prediction results: {e}", exc_info=True)
            return f"Error: 解析多产物预测结果时出错: {str(e)}"

    except Exception as e:
        logger.error(f"Error in predict_reaction_topn: {e}")
        return f"Error: {str(e)}"





@llm_tool(name="predict_reaction_properties_rxn",
          description="Predict chemical reaction properties such as atom mapping and yield using IBM RXN for Chemistry API")
async def predict_reaction_properties_rxn(
    reaction: str,
    property_type: str = "atom-mapping"
) -> str:
    """
    Predict chemical reaction properties such as atom mapping and yield.

    This function uses the IBM RXN for Chemistry API to predict various properties
    of chemical reactions, including atom-to-atom mapping (showing how atoms in
    reactants correspond to atoms in products) and potential reaction yields.

    Args:
        reaction: SMILES notation of the reaction (reactants>>products).
        property_type: Type of property to predict. Options: "atom-mapping", "yield".

    Returns:
        Formatted Markdown string containing predicted reaction properties.

    Examples:
        >>> predict_reaction_properties_rxn("CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F", "atom-mapping")
        # Returns atom mapping for the reaction
    """
    try:
        # Get RXN wrapper
        wrapper = _get_rxn_wrapper()

        # Clean input
        reaction = reaction.strip()

        # Validate property_type
        valid_property_types = ["atom-mapping", "yield"]
        if property_type not in valid_property_types:
            return f"Error: 无效的属性类型 '{property_type}'。支持的类型: {', '.join(valid_property_types)}"

        # Determine model based on property type
        ai_model = "atom-mapping-2020" if property_type == "atom-mapping" else "yield-2020-08-10"

        # Submit prediction
        response = await asyncio.to_thread(
            wrapper.predict_reaction_properties,
            reactions=[reaction],
            ai_model=ai_model
        )

        if not response or "response" not in response:
            return f"Error: 无法提交{property_type}预测请求"

        # Extract results
        try:
            content = response.get("response", {}).get("payload", {}).get("content", [])

            if not content:
                return f"Error: 未找到{property_type}预测结果"

            # Format results based on property type
            markdown = f"## 反应{property_type}预测结果\n\n"
            markdown += f"### 输入反应\n\n`{reaction}`\n\n"

            if property_type == "atom-mapping":
                # Extract mapped reaction
                mapped_reaction = content[0].get("value", "")

                if not mapped_reaction:
                    return "Error: 无法生成原子映射"

                markdown += "### 原子映射结果\n\n"
                markdown += f"`{mapped_reaction}`\n\n"

                # Split into reactants and products for explanation
                if ">>" in mapped_reaction:
                    reactants, products = mapped_reaction.split(">>")
                    markdown += "### 映射解释\n\n"
                    markdown += "原子映射显示了反应物中的原子如何对应到产物中的原子。\n"
                    markdown += "每个原子上的数字表示映射ID，相同ID的原子在反应前后是同一个原子。\n\n"
                    markdown += f"**映射的反应物:** `{reactants}`\n\n"
                    markdown += f"**映射的产物:** `{products}`\n"

            elif property_type == "yield":
                # Extract predicted yield
                predicted_yield = content[0].get("value", "")

                if not predicted_yield:
                    return "Error: 无法预测反应产率"

                try:
                    yield_value = float(predicted_yield)
                    markdown += "### 产率预测结果\n\n"
                    markdown += f"**预测产率:** {yield_value:.1f}%\n\n"

                    # Add interpretation
                    if yield_value < 30:
                        markdown += "**解释:** 预测产率较低，反应可能效率不高。考虑优化反应条件或探索替代路线。\n"
                    elif yield_value < 70:
                        markdown += "**解释:** 预测产率中等，反应可能是可行的，但有优化空间。\n"
                    else:
                        markdown += "**解释:** 预测产率较高，反应可能非常有效。\n"
                except ValueError:
                    markdown += f"**预测产率:** {predicted_yield}\n"

            return markdown

        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing reaction properties results: {e}")
            return f"Error: 解析反应属性预测结果时出错: {str(e)}"

    except Exception as e:
        logger.error(f"Error in predict_reaction_properties: {e}")
        return f"Error: {str(e)}"


def collect_reactions(tree):
    """
    从逆合成树中收集反应SMILES。

    Args:
        tree: 逆合成树节点

    Returns:
        反应SMILES列表
    """
    reactions = []
    if 'children' in tree and tree['children']:
        reaction_smarts = '{}>>{}'.format(
            '.'.join([node['smiles'] for node in tree['children']]),
            tree['smiles']
        )
        reactions.append(reaction_smarts)
    for node in tree.get('children', []):
        reactions.extend(collect_reactions(node))
    return reactions


@llm_tool(name="extract_reaction_actions_rxn",
          description="Extract structured reaction steps from text descriptions using IBM RXN for Chemistry API")
async def extract_reaction_actions_rxn(reaction_text: str) -> str:
    """
    Extract structured reaction steps from text descriptions.

    This function uses the IBM RXN for Chemistry API to parse text descriptions
    of chemical procedures and extract structured actions representing the steps
    of the procedure.

    Args:
        reaction_text: Text description of a chemical reaction procedure.

    Returns:
        Formatted Markdown string containing the extracted reaction steps.

    Examples:
        >>> extract_reaction_actions_rxn("To a stirred solution of 7-(difluoromethylsulfonyl)-4-fluoro-indan-1-one (110 mg, 0.42 mmol) in methanol (4 mL) was added sodium borohydride (24 mg, 0.62 mmol). The reaction mixture was stirred at ambient temperature for 1 hour.")
        # Returns structured steps extracted from the text
    """
    try:
        # Get RXN wrapper
        wrapper = _get_rxn_wrapper()

        # Clean input
        reaction_text = reaction_text.strip()

        if not reaction_text:
            return "Error: 反应文本为空"

        # Submit extraction request
        response = await asyncio.to_thread(
            wrapper.paragraph_to_actions,
            paragraph=reaction_text
        )

        # 检查response是否存在
        if not response:
            return "Error: 无法从文本中提取反应步骤"

        # 直接返回response，不做任何处理
        # 这是基于参考代码中直接打印results['actions']的方式
        # 我们假设response本身就是我们需要的结果
        return f"""## 反应步骤提取结果

### 输入文本

{reaction_text}

### 提取的反应步骤

```
{response}
```
"""



    except Exception as e:
        logger.error(f"Error in extract_reaction_actions: {e}")
        return f"Error: {str(e)}"


@llm_tool(name="predict_retrosynthesis_rxn",
          description="预测目标分子的逆合成路径，包括详细的反应SMILES")
async def predict_retrosynthesis_rxn(target_molecule: str, max_steps: int = 3) -> str:
    """
    预测目标分子的逆合成路径，并返回详细的反应SMILES。

    此函数使用IBM RXN for Chemistry API建议可能的合成路线，
    将目标分子分解为可能商业可得的更简单前体，并提供每个步骤的详细反应SMILES。

    Args:
        target_molecule: 目标分子的SMILES表示法。
        max_steps: 考虑的最大逆合成步骤数，默认为3。

    Returns:
        包含预测逆合成路径和详细反应SMILES的格式化Markdown字符串。

    Examples:
        >>> predict_retrosynthesis_rxn("Brc1c2ccccc2c(Br)c2ccccc12")
        # 返回目标分子的可能合成路线及详细反应SMILES
    """
    try:
        # Get RXN wrapper
        wrapper = _get_rxn_wrapper()

        # Clean input
        target_molecule = target_molecule.strip()

        # Validate max_steps
        if max_steps < 1:
            max_steps = 1
        elif max_steps > 5:
            max_steps = 5
            logger.warning("max_steps限制为最大5步")

        # Submit prediction
        response = await asyncio.to_thread(
            wrapper.predict_automatic_retrosynthesis,
            product=target_molecule,
            max_steps=max_steps
        )

        if not response or "prediction_id" not in response:
            return "Error: 无法提交逆合成预测请求"

        # 获取结果
        prediction_id = response['prediction_id']
        max_retries = 10
        retries = 0

        while retries < max_retries:
            results = await asyncio.to_thread(
                wrapper.get_predict_automatic_retrosynthesis_results,
                prediction_id
            )

            status = results.get('status', 'PENDING')

            if status == 'SUCCESS':
                break
            elif status in ['NEW', 'PENDING', 'PROCESSING']:
                await asyncio.sleep(15)  # 等待15秒再检查
                retries += 1
            else:
                error_message = results.get('errorMessage', '未知错误')
                return f"Error: 预测失败，状态: {status}。错误信息: {error_message}"

        if retries >= max_retries:
            return "Error: 预测过程超过最大重试次数或超时"

        # Extract retrosynthetic paths
        try:
            paths = results.get("retrosynthetic_paths", [])

            if not paths:
                return "## 逆合成分析结果\n\n未找到可行的逆合成路径。目标分子可能太复杂或结构有问题。"

            # Format results
            markdown = f"## 逆合成分析结果\n\n"
            markdown += f"### 目标分子\n\n`{target_molecule}`\n\n"
            markdown += f"### 找到的合成路径: {len(paths)}\n\n"

            # Limit to top 3 paths for readability
            display_paths = paths[:3]

            for i, path in enumerate(display_paths, 1):
                markdown += f"#### 路径 {i}\n\n"

                # Extract sequence information
                sequence_id = path.get("sequenceId", "未知")
                confidence = path.get("confidence", 0.0)

                markdown += f"**置信度:** {confidence:.2f}\n\n"
                markdown += f"**路径ID:** `{sequence_id}`\n\n"

                # 收集并显示反应SMILES
                reactions = collect_reactions(path)
                if reactions:
                    markdown += "**反应SMILES:**\n\n"
                    for j, reaction in enumerate(reactions, 1):
                        markdown += f"{j}. `{reaction}`\n"
                    markdown += "\n"

                # Extract steps
                steps = path.get("steps", [])

                if steps:
                    markdown += "**合成步骤:**\n\n"

                    for j, step in enumerate(steps, 1):
                        # Extract reactants and products
                        reactants = step.get("reactants", [])
                        reactant_smiles = [r.get("smiles", "") for r in reactants if "smiles" in r]

                        product = step.get("product", {})
                        product_smiles = product.get("smiles", "")

                        markdown += f"步骤 {j}: "

                        if reactant_smiles and product_smiles:
                            markdown += f"`{'.' if len(reactant_smiles) > 1 else ''.join(reactant_smiles)}` → `{product_smiles}`\n\n"
                        else:
                            markdown += "反应细节不可用\n\n"
                else:
                    markdown += "**合成步骤:** 未提供详细步骤\n\n"

                markdown += "---\n\n"

            if len(paths) > 3:
                markdown += f"*注: 仅显示前3条路径，共找到{len(paths)}条可能的合成路径。*\n"

            return markdown

        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing retrosynthesis results: {e}")
            return f"Error: 解析逆合成结果时出错: {str(e)}"

    except Exception as e:
        logger.error(f"Error in predict_retrosynthesis: {e}")
        return f"Error: {str(e)}"


def collect_reactions(tree):
    """
    从逆合成树中收集反应SMILES。

    Args:
        tree: 逆合成树节点

    Returns:
        反应SMILES列表
    """
    reactions = []
    if 'children' in tree and tree['children']:
        reaction_smarts = '{}>>{}'.format(
            '.'.join([node['smiles'] for node in tree['children']]),
            tree['smiles']
        )
        reactions.append(reaction_smarts)
    for node in tree.get('children', []):
        reactions.extend(collect_reactions(node))
    return reactions

@llm_tool(name="create_and_start_synthesis_rxn",
          description="从逆合成序列创建和启动合成计划")
async def create_and_start_synthesis_rxn(sequence_id: str) -> str:
    """
    从给定的逆合成序列ID创建合成计划并启动它。

    此函数使用IBM RXN for Chemistry API从逆合成分析结果创建合成计划，
    并返回合成状态和计划详情。

    Args:
        sequence_id: 来自逆合成路径的序列ID。

    Returns:
        包含合成状态和计划的格式化Markdown字符串。

    Examples:
        >>> create_and_start_synthesis_rxn("rp-1234567890")
        # 返回从序列ID创建的合成计划和状态
    """
    try:
        # Get RXN wrapper
        wrapper = _get_rxn_wrapper()

        # Clean input
        sequence_id = sequence_id.strip()

        # Create synthesis from sequence
        response = await asyncio.to_thread(
            wrapper.create_synthesis_from_sequence,
            sequence_id
        )

        if not response or "synthesis_id" not in response:
            return "Error: 无法从序列创建合成计划"

        synthesis_id = response['synthesis_id']

        # 获取节点ID
        node_ids = await asyncio.to_thread(
            wrapper.get_node_ids,
            synthesis_id
        )

        if not node_ids:
            return "Error: 节点ID列表为空"

        # 初始化操作列表
        ordered_list_of_actions = []

        # 对每个节点ID获取反应设置
        for node_id in node_ids:
            reaction_settings = await asyncio.to_thread(
                wrapper.get_reaction_settings,
                synthesis_id,
                node_id
            )

            if reaction_settings:
                # 提取该节点的操作并添加到列表
                actions = reaction_settings.get('actions', [])
                ordered_list_of_actions.extend(actions)

        # Start synthesis
        status = await asyncio.to_thread(
            wrapper.start_synthesis,
            synthesis_id
        )

        # Format results
        markdown = f"## 合成计划创建与启动\n\n"
        markdown += f"### 序列ID\n\n`{sequence_id}`\n\n"
        markdown += f"### 合成ID\n\n`{synthesis_id}`\n\n"

        if ordered_list_of_actions:
            markdown += "### 合成步骤\n\n"
            for i, action in enumerate(ordered_list_of_actions, 1):
                markdown += f"{i}. `{action}`\n"  # 可以只取name 字段， content 字段还有更多详细信息（dict）
            markdown += "\n"
        else:
            markdown += "### 合成步骤\n\n未提供详细步骤\n\n"

        markdown += f"### 合成状态\n\n"
        markdown += f"**状态:** `{status.get('status', '未知')}`\n\n"

        return markdown

    except Exception as e:
        logger.error(f"Error in create_and_start_synthesis: {e}")
        return f"Error: {str(e)}"
