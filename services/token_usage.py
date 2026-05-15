from dataclasses import dataclass
from typing import Any


@dataclass(eq=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    analysis_tokens: int = 0


def _read_path(data: Any, *path: str):
    current = data
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
            continue
        current = getattr(current, key, None)
    return current


def _first_present(response: Any, *paths: tuple[str, ...]):
    for path in paths:
        value = _read_path(response, *path)
        if value is not None:
            return value
    return 0


def _to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def normalize_token_usage(response: Any) -> TokenUsage:
    input_tokens = _first_present(
        response,
        ("usage_metadata", "input_tokens"),
        ("response_metadata", "token_usage", "prompt_tokens"),
        ("token_usage", "prompt_tokens"),
        ("response_metadata", "input_tokens"),
    )
    output_tokens = _first_present(
        response,
        ("usage_metadata", "output_tokens"),
        ("response_metadata", "token_usage", "completion_tokens"),
        ("token_usage", "completion_tokens"),
        ("response_metadata", "output_tokens"),
    )
    analysis_tokens = _first_present(
        response,
        ("usage_metadata", "reasoning_tokens"),
        ("response_metadata", "reasoning_tokens"),
        ("response_metadata", "output_tokens_details", "reasoning_tokens"),
        ("response_metadata", "token_usage", "output_tokens_details", "reasoning_tokens"),
    )
    return TokenUsage(
        input_tokens=_to_int(input_tokens),
        output_tokens=_to_int(output_tokens),
        analysis_tokens=_to_int(analysis_tokens),
    )


def merge_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        analysis_tokens=left.analysis_tokens + right.analysis_tokens,
    )


def usage_dict(usage: TokenUsage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "analysis_tokens": usage.analysis_tokens,
    }