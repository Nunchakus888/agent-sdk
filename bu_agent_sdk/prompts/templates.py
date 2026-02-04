"""
共享模板常量

供 ConfigEnhancer 复用。
"""

# Instructions 结构模板 - 定义 instructions 的标准格式
INSTRUCTIONS_TEMPLATE = """# ROLE & IDENTITY
You are {name}, {description}.

## BACKGROUND
{background}

## WORKFLOW PROCEDURE
{workflow_steps}

## BOUNDARIES & CONSTRAINTS
{constraints}
"""

# Knowledge Base Section - 运行时动态注入 retrieval 结果
# 使用方式: KNOWLEDGE_BASE_SECTION.format(kb_content=retrieval_results)
KNOWLEDGE_BASE_SECTION = """## Reference Knowledge
The following information has been retrieved from the knowledge base based on the conversation context. Use this as reference when responding:

{kb_content}

Note: This knowledge is provided as context. Prioritize this information when answering related questions."""

__all__ = [
    "INSTRUCTIONS_TEMPLATE",
    "KNOWLEDGE_BASE_SECTION",
]
