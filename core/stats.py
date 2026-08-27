"""
Token 用量提取 & 调用统计。

供推理与流式路径复用。
"""


def _extract_token_usage(response) -> dict[str, int]:
    """
    从 LangChain AIMessage / BaseMessage 中提取 token 用量。

    兼容多种响应格式（usage_metadata / response_metadata / additional_kwargs）。
    返回 {"input_tokens": N, "output_tokens": N, "total_tokens": N}，提取失败时均为 0。
    """
    usage: dict[str, int] = {}

    # 1) usage_metadata（LangChain >= 0.3 标准属性）
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        if isinstance(um, dict):
            usage['input_tokens'] = int(um.get('input_tokens', 0))
            usage['output_tokens'] = int(um.get('output_tokens', 0))
            usage['total_tokens'] = int(um.get('total_tokens', 0))
            if usage['total_tokens']:
                return usage

    # 2) response_metadata（SiliconFlow / OpenAI 兼容格式）
    if hasattr(response, 'response_metadata'):
        rm = response.response_metadata or {}
        for key in ('token_usage', 'usage'):
            tu = rm.get(key)
            if isinstance(tu, dict):
                usage['input_tokens'] = int(tu.get('prompt_tokens', 0))
                usage['output_tokens'] = int(tu.get('completion_tokens', 0))
                usage['total_tokens'] = int(tu.get('total_tokens', 0))
                if usage['total_tokens']:
                    return usage

    # 3) additional_kwargs（某些旧版本 / 自定义 provider）
    if hasattr(response, 'additional_kwargs'):
        ak = response.additional_kwargs or {}
        for key in ('token_usage', 'usage'):
            tu = ak.get(key)
            if isinstance(tu, dict):
                usage['input_tokens'] = int(tu.get('prompt_tokens', 0))
                usage['output_tokens'] = int(tu.get('completion_tokens', 0))
                usage['total_tokens'] = int(tu.get('total_tokens', 0))
                if usage['total_tokens']:
                    return usage

    # 4) 顶层属性（某些封装会把 usage 直接挂在 message 上）
    for attr in ('token_usage', 'usage', 'usage_metadata'):
        tu = getattr(response, attr, None)
        if isinstance(tu, dict):
            usage['input_tokens'] = int(tu.get('prompt_tokens', tu.get('input_tokens', 0)))
            usage['output_tokens'] = int(tu.get('completion_tokens', tu.get('output_tokens', 0)))
            usage['total_tokens'] = int(tu.get('total_tokens', 0))
            if usage['total_tokens']:
                return usage

    return usage


def _extract_token_usage_from_text(text_length: int) -> dict:
    """根据文本长度粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）。"""
    estimated = max(1, int(text_length / 2.5))
    return {"input_tokens": 0, "output_tokens": estimated, "total_tokens": estimated}


class _CallStats:
    """LLM 调用统计累加器（线程安全仅限单事件循环）。"""

    def __init__(self, label: str = ""):
        self.label = label
        self.total_calls: int = 0
        self.total_elapsed: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_failures: int = 0

    def record(self, elapsed: float, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.total_calls += 1
        self.total_elapsed += elapsed
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_failure(self) -> None:
        self.total_failures += 1

    def summary(self) -> str:
        parts = [f"[{self.label}] calls={self.total_calls}"]
        if self.total_failures:
            parts.append(f"fail={self.total_failures}")
        if self.total_calls > 0:
            avg = self.total_elapsed / self.total_calls
            parts.append(f"total={self.total_elapsed:.1f}s avg={avg:.1f}s/call")
            parts.append(f"in_tok={self.total_input_tokens} out_tok={self.total_output_tokens}")
        else:
            parts.append("(no successful calls)")
        return ", ".join(parts)

    def reset(self) -> None:
        self.total_calls = 0
        self.total_elapsed = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0


_text_stats = _CallStats("text")
_vision_stats = _CallStats("vision")


# 命名累加器注册表：供 embedding / rerank 等非 LLM 调用路径复用 _CallStats。
# 按名字懒创建并缓存，便于统一观测；LLM 的 text/vision 仍走上方显式单例。
_registry: dict[str, _CallStats] = {}


def get_stat(name: str) -> _CallStats:
    """按名获取（懒创建）共享累加器。"""
    stat = _registry.get(name)
    if stat is None:
        stat = _CallStats(name)
        _registry[name] = stat
    return stat


def _snapshot(stat: _CallStats) -> dict:
    """单个累加器的可序列化摘要。"""
    return {
        "calls": stat.total_calls,
        "total_elapsed_s": round(stat.total_elapsed, 3),
        "total_input_tokens": stat.total_input_tokens,
        "total_output_tokens": stat.total_output_tokens,
    }


def get_all_stats() -> dict:
    """返回全部已注册累加器（含 LLM text/vision）的累计统计。"""
    result = {
        "text": _snapshot(_text_stats),
        "vision": _snapshot(_vision_stats),
    }
    for name, stat in _registry.items():
        result[name] = _snapshot(stat)
    return result


def get_llm_stats() -> dict:
    """返回文本 LLM 和 VLM 的累计调用统计（向后兼容，仅含 LLM 两项）。"""
    return {
        "text": _snapshot(_text_stats),
        "vision": _snapshot(_vision_stats),
    }