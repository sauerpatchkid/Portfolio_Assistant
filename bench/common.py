# --- Shared Benchmark Helpers ---
import statistics
import time


async def time_first_token(client, model, messages, tools=None) -> dict:
    """
    Time-to-first-token for one request, measured as the latency of a request
    capped at ONE output token: the server has to process the whole prompt
    (prefill) and produce a single token, which is what TTFT measures.
    Works the same on any OpenAI-compatible server.
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if tools:
        kwargs["tools"] = tools

    start = time.perf_counter()
    response = await client.chat.completions.create(**kwargs)
    ttft = time.perf_counter() - start

    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "ttft_s": round(ttft, 4),
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
    }


def median(values):
    return round(statistics.median(values), 4) if values else None


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return round(ordered[index], 4)
