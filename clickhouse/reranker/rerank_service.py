"""Reranker service for the topic-curation tool (D-097, B72): Qwen3-Reranker-4B (Q4_K_M, llama.cpp), loaded once, one request at a time.

Same recipe as B62/B63 (out/b62/b62_etapa1.py): prefix (system + instruction 1 + query) evaluated once per request and its KV cache reused for every
document; score = P("yes") from the "yes"/"no" logits at the last position; document = "{name}. {keywords}. {summary}" (built by quyca).
Listens on 127.0.0.1 only. API (internal, not the public contract):
  GET  /health  -> {"status": "ok", "model": ..., "queue": n, "busy": bool}
  POST /rerank  {"seed_text": str, "documents": [{"topic_id": str, "text": str}], "timeout_s": float}
                -> {"results": [{"topic_id": str, "score": float}, ...] best first, "seconds": float, "seed_truncated": bool}
Requests go through a FIFO queue served by one worker thread; a request that waits longer than timeout_s gets HTTP 503 and is dropped before it starts.
No request text is logged. Configuration by environment variables (defaults in brackets): RERANKER_GGUF [~/ASCUN/modelos/qwen3-reranker-4b-q4_k_m.gguf],
RERANKER_PORT [11437], RERANKER_NGL [99 = all layers on the GPU], RERANKER_THREADS [6], RERANKER_CTX [3072]."""
import json
import math
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

GGUF = os.environ.get("RERANKER_GGUF", os.path.expanduser("~/ASCUN/modelos/qwen3-reranker-4b-q4_k_m.gguf"))
PORT = int(os.environ.get("RERANKER_PORT", "11437"))
NGL = int(os.environ.get("RERANKER_NGL", "99"))
THREADS = int(os.environ.get("RERANKER_THREADS", "6"))
CTX = int(os.environ.get("RERANKER_CTX", "3072"))
MAX_SEED_TOKENS = 2000  # longer seeds are cut for the reranker (the cut is reported); the rest of the context is for one document
MAX_DOC_TOKENS = 700
INSTRUCTION = "Given a thematic axis description, judge whether the research topic belongs to the axis"  # instruction 1 of B62
PRE = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Scorer:
    def __init__(self, gguf: str, ngl: int, threads: int, ctx: int) -> None:
        from llama_cpp import Llama

        self.llm = Llama(model_path=gguf, n_ctx=ctx, n_threads=threads, n_gpu_layers=ngl, verbose=False, logits_all=False)
        self.yes, self.no = self.llm.tokenize(b"yes", add_bos=False)[0], self.llm.tokenize(b"no", add_bos=False)[0]
        self.plen = 0

    def set_query(self, seed: str) -> bool:
        ids = self.llm.tokenize(seed.encode(), add_bos=False)
        cut = len(ids) > MAX_SEED_TOKENS
        if cut:
            seed = self.llm.detokenize(ids[:MAX_SEED_TOKENS]).decode("utf-8", "ignore")
        head = PRE + "<Instruct>: %s\n<Query>: %s\n<Document>:" % (INSTRUCTION, seed)
        pt = self.llm.tokenize(head.encode(), add_bos=False, special=True)
        self.llm.reset()
        self.llm.eval(pt)
        self.plen = len(pt)
        return cut

    def one(self, doc: str) -> float:
        dt = self.llm.tokenize((" " + doc).encode(), add_bos=False, special=True)[:MAX_DOC_TOKENS] + self.llm.tokenize(SUF.encode(), add_bos=False, special=True)
        self.llm._ctx.kv_cache_seq_rm(0, self.plen, -1)  # drop the previous document, keep the prefix
        self.llm.n_tokens = self.plen
        self.llm.eval(dt)
        lg = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm._n_vocab,))
        return 1 / (1 + math.exp(float(lg[self.no]) - float(lg[self.yes])))


class Job:
    def __init__(self, seed: str, docs: list, deadline: float) -> None:
        self.seed, self.docs, self.deadline = seed, docs, deadline
        self.done = threading.Event()
        self.result: dict | None = None
        self.error: str | None = None
        self.cancelled = False


JOBS: "queue.Queue[Job]" = queue.Queue()  # FIFO
STATE = {"busy": False, "scorer": None}


def worker() -> None:
    scorer: Scorer = STATE["scorer"]  # type: ignore[assignment]
    while True:
        job = JOBS.get()
        if job.cancelled or time.monotonic() > job.deadline:
            job.error = "deadline exceeded while queued"
            job.done.set()
            continue
        STATE["busy"] = True
        t0 = time.monotonic()
        try:
            cut = scorer.set_query(job.seed)
            scored = [(d["topic_id"], scorer.one(d["text"])) for d in job.docs]
            scored.sort(key=lambda x: -x[1])
            job.result = {"results": [{"topic_id": t, "score": round(s, 6)} for t, s in scored], "seconds": round(time.monotonic() - t0, 2), "seed_truncated": cut}
        except Exception as e:  # reported to the caller as 503; the worker stays alive
            job.error = "%s: %s" % (type(e).__name__, e)
        finally:
            STATE["busy"] = False
            job.done.set()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": os.path.basename(GGUF), "ngl": NGL, "queue": JOBS.qsize(), "busy": STATE["busy"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/rerank":
            return self._send(404, {"error": "not found"})
        t_arrival = time.monotonic()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            seed, docs = body["seed_text"], body["documents"]
            timeout_s = float(body.get("timeout_s", 60))
            if not isinstance(seed, str) or not seed or not isinstance(docs, list) or not docs or not all(isinstance(d, dict) and "topic_id" in d and "text" in d for d in docs):
                raise ValueError("bad payload")
        except (KeyError, ValueError, TypeError):
            return self._send(400, {"error": "seed_text and documents [{topic_id, text}] are required"})
        job = Job(seed, docs, t_arrival + timeout_s)
        JOBS.put(job)
        if not job.done.wait(timeout=timeout_s):
            job.cancelled = True  # dropped if it has not started; if it is running, its result is discarded
            return self._send(503, {"error": "timeout"})
        if job.error or job.result is None:
            return self._send(503, {"error": job.error or "failed"})
        self._send(200, job.result)

    def log_message(self, fmt: str, *args: object) -> None:  # request text is never logged
        pass


def main() -> None:
    t0 = time.time()
    STATE["scorer"] = Scorer(GGUF, NGL, THREADS, CTX)
    print("model loaded in %.1f s: %s (n_gpu_layers=%d)" % (time.time() - t0, os.path.basename(GGUF), NGL), flush=True)
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("listening on 127.0.0.1:%d" % PORT, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
