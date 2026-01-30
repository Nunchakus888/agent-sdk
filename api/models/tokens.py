from dataclasses import dataclass
from typing import Mapping, Optional

from openai.types.completion_usage import PromptTokensDetails, CompletionTokensDetails


@dataclass(frozen=True)
class UsageInfo:
    input_tokens: int
    output_tokens: int
    total_tokens: int | None = 0
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None


@dataclass(frozen=True)
class GenerationInfo:
    schema_name: str
    model: str
    duration: float
    usage: UsageInfo | None = None
