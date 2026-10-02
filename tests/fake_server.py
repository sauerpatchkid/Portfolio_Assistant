# --- Fake OpenAI-Compatible Server ---
# A stand-in for vLLM / SGLang that needs no GPU. It answers every eval question
# perfectly by following the question's oracle, so the command-line scripts can
# be exercised end to end on a laptop:
#
#   python -m tests.fake_server --port 8000
#   python -m eval.run_eval --name fake --model fake --concurrency 16
#   python -m bench.ttft --name fake_ttft --model fake --limit 10
import argparse
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import eval.offline as offline

from tests.fakes import OracleBackend


class Handler(BaseHTTPRequestHandler):
    oracle = None

    def log_message(self, *args):
        pass

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_json({"status": "ok"})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = request["messages"]

        known = any(m["role"] == "user" and m["content"] in self.oracle.by_first_message for m in messages)
        if known and request.get("max_tokens", 512) > 1:
            turn = asyncio.run(self.oracle.chat(messages))
            message = {"role": "assistant", "content": turn.content}
            if turn.tool_calls:
                message["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in turn.tool_calls
                ]
        else:
            message = {"role": "assistant", "content": "OK"}

        prompt_tokens = len(json.dumps(messages)) // 4
        self.send_json({
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": request["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1,
                      "total_tokens": prompt_tokens + 1},
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    Handler.oracle = OracleBackend(offline.load_jsonl(offline.QUESTIONS_PATH)
                                   + offline.load_jsonl(offline.HOLDOUT_PATH))
    print(f"Fake server on http://localhost:{args.port}/v1")
    ThreadingHTTPServer(("localhost", args.port), Handler).serve_forever()
