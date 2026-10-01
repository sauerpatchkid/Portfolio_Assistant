# --- Model Backends ---
# The agent loop only needs one thing from a model: "given these messages and
# tools, give me the next assistant turn". Two backends provide that:
#   OpenAIBackend - any OpenAI-compatible server (vLLM, vLLM + LMCache, SGLang)
#   HFBackend     - in-process Hugging Face generate(), the original notebook path,
#                   kept as the sequential baseline
import json
import re
import time
from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    content: str
    tool_calls: list = field(default_factory=list)   # [{"id", "name", "arguments": dict}]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0


def strip_think_tags(text):
    """Remove Qwen <think>...</think> blocks (safety net if any slip through)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_tool_calls(response_text: str) -> list[dict]:
    """Parse Hermes-style <tool_call>...</tool_call> blocks out of raw model text."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    tool_calls = []
    for i, match in enumerate(matches):
        try:
            parsed = json.loads(match.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        arguments = parsed.get("arguments", {})
        tool_calls.append({
            "id": f"call_{i}",
            "name": parsed.get("name", ""),
            "arguments": arguments if isinstance(arguments, dict) else {},
        })
    return tool_calls


def strip_tool_call_blocks(text: str) -> str:
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()


class OpenAIBackend:
    """Talks to an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, base_url, model, api_key="EMPTY", timeout=600):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def chat(self, messages, tools=None, max_tokens=512) -> ChatTurn:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,   # deterministic for reproducible testing
            # non-thinking mode for cleaner, faster output
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        response = await self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start

        message = response.choices[0].message
        content = strip_think_tags(message.content or "")

        tool_calls = []
        for tc in message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            })

        # If the server's tool-call parser missed a call, it shows up as raw text.
        if not tool_calls and "<tool_call>" in content:
            tool_calls = parse_tool_calls(content)
            content = strip_tool_call_blocks(content)

        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)

        return ChatTurn(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            latency_s=latency,
        )


class HFBackend:
    """In-process Hugging Face generation, one request at a time."""

    def __init__(self, model_id, quantized=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        if quantized:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    async def chat(self, messages, tools=None, max_tokens=512) -> ChatTurn:
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        latency = time.perf_counter() - start

        new_tokens = outputs[0][prompt_tokens:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        tool_calls = parse_tool_calls(response_text)
        content = strip_think_tags(strip_tool_call_blocks(response_text))

        return ChatTurn(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(new_tokens),
            latency_s=latency,
        )
