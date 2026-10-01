# --- Start / Stop a Model Server From a Notebook ---
# The server commands themselves live in the serving/*.sh scripts. This helper
# runs one of them in the background, waits until the server answers, and
# shuts it down cleanly so the next configuration can use the GPU.
#
#   server = start_server("vllm", "Qwen/Qwen3-8B-AWQ")
#   ...run evals / benchmarks against http://localhost:8000/v1...
#   stop_server(server)
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "logs"


def _is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def tail(path, lines=60) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-lines:])


def start_server(config: str, model: str, port: int = 8000, extra_args: str = "",
                 env: dict = None, timeout_s: int = 1200, log_name: str = None):
    """
    Run serving/<config>.sh in the background and wait until it serves requests.
    Returns the process; its log path is at process.log_path.
    """
    script = ROOT / "serving" / f"{config}.sh"
    if not script.exists():
        raise ValueError(f"No serving script named {script.name}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name or config}.log"

    process_env = {**os.environ, **(env or {}), "EXTRA_ARGS": extra_args}
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        ["bash", str(script), model, str(port)],
        stdout=log_file, stderr=subprocess.STDOUT,
        env=process_env, cwd=ROOT,
        start_new_session=True,   # own process group, so stop_server can end every child
    )
    process.log_path = log_path

    print(f"Starting {config} with {model} (log: {log_path}) ...")
    start = time.time()
    while time.time() - start < timeout_s:
        if process.poll() is not None:
            print(tail(log_path))
            raise RuntimeError(f"{config} exited during start-up (code {process.returncode}). Log tail is above.")
        if _is_ready(port):
            print(f"Ready after {time.time() - start:.0f}s")
            return process
        time.sleep(5)

    stop_server(process)
    print(tail(log_path))
    raise RuntimeError(f"{config} was not ready after {timeout_s}s. Log tail is above.")


def run_command(command: str) -> int:
    """Run a shell command and print its output as it arrives (for use inside notebook functions)."""
    process = subprocess.Popen(
        command, shell=True, cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in process.stdout:
        print(line, end="")
    return process.wait()


def stop_server(process, wait_s: int = 30):
    """Stop the server and everything it started, then give the GPU a moment to free memory."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass
    time.sleep(10)
    print("Server stopped.")
