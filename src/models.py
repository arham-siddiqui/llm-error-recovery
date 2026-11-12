"""LLM client abstraction with on-disk caching, retries and timeouts.

Backends: ollama, openai, openai_compatible, anthropic, mock (unit tests ONLY).
API keys are read from environment variables named in the config and are never logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import requests


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    cached: bool = False
    backend_meta: dict = field(default_factory=dict)


class ResponseCache:
    """Append-only JSONL cache. Doubles as a checkpoint: every completed call is fsynced."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated final line after a crash
                    self.data[rec["key"]] = rec["response"]

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, resp: dict) -> None:
        with self.lock:
            self.data[key] = resp
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "response": resp}) + "\n")
                f.flush()
                os.fsync(f.fileno())


class LLMClient:
    backend = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024, seed: int = 0,
                 timeout_s: float = 180, max_retries: int = 3, cache: ResponseCache | None = None,
                 name: str | None = None, input_cost_per_mtok: float = 0.0,
                 output_cost_per_mtok: float = 0.0, **extra: Any):
        self.model = model
        self.name = name or model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache = cache
        self.input_cost_per_mtok = input_cost_per_mtok
        self.output_cost_per_mtok = output_cost_per_mtok
        self.extra = extra

    # Everything that affects the output goes into the cache key. No secrets.
    def cache_config(self) -> dict:
        return {"backend": self.backend, "model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "seed": self.seed,
                **{k: v for k, v in self.extra.items() if k in ("num_ctx", "base_url")}}

    def describe(self) -> dict:
        """Exact model identifier + generation config for the run manifest."""
        return {"name": self.name, **self.cache_config()}

    def estimate_cost(self, in_tok: int | None, out_tok: int | None) -> float:
        return ((in_tok or 0) * self.input_cost_per_mtok + (out_tok or 0) * self.output_cost_per_mtok) / 1e6

    def _call(self, messages: list[dict]) -> LLMResponse:
        raise NotImplementedError

    def chat(self, messages: list[dict]) -> LLMResponse:
        key = hashlib.sha256(json.dumps({"cfg": self.cache_config(), "messages": messages},
                                        sort_keys=True).encode()).hexdigest()
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return LLMResponse(**{**hit, "cached": True})
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._call(messages)
                break
            except Exception as e:  # noqa: BLE001 - retry on any transport/API error
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        else:
            raise RuntimeError(f"{self.backend} call failed after retries: {type(last_err).__name__}: "
                               f"{str(last_err)[:200]}")
        if self.cache is not None:
            d = asdict(resp)
            d["cached"] = False
            self.cache.put(key, d)
        return resp


class OllamaClient(LLMClient):
    backend = "ollama"

    def __init__(self, *a, base_url: str = "http://localhost:11434", num_ctx: int = 8192, **kw):
        super().__init__(*a, base_url=base_url, num_ctx=num_ctx, **kw)
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx

    def describe(self) -> dict:
        d = super().describe()
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=10).json()
            for m in tags.get("models", []):
                if m["name"] == self.model:
                    d["digest"] = m.get("digest")
                    d["details"] = m.get("details")
        except Exception:  # noqa: BLE001
            pass
        return d

    def _call(self, messages):
        t0 = time.time()
        r = requests.post(f"{self.base_url}/api/chat", timeout=self.timeout_s, json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": self.max_tokens, "num_ctx": self.num_ctx}})
        r.raise_for_status()
        d = r.json()
        return LLMResponse(text=d["message"]["content"], model=d.get("model", self.model),
                           input_tokens=d.get("prompt_eval_count"), output_tokens=d.get("eval_count"),
                           latency_s=time.time() - t0, finish_reason=d.get("done_reason"))


class OpenAIClient(LLMClient):
    """OpenAI API, or any OpenAI-compatible endpoint (vLLM, Together, OpenRouter...) via base_url."""
    backend = "openai"

    def __init__(self, *a, api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None, **kw):
        super().__init__(*a, base_url=base_url, **kw)
        import openai
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        self._client = openai.OpenAI(api_key=key, base_url=base_url, timeout=self.timeout_s, max_retries=0)
        if base_url:
            self.backend = "openai_compatible"

    def _call(self, messages):
        t0 = time.time()
        r = self._client.chat.completions.create(model=self.model, messages=messages,
                                                 temperature=self.temperature,
                                                 max_tokens=self.max_tokens, seed=self.seed)
        u = r.usage
        return LLMResponse(text=r.choices[0].message.content or "", model=r.model,
                           input_tokens=getattr(u, "prompt_tokens", None),
                           output_tokens=getattr(u, "completion_tokens", None),
                           latency_s=time.time() - t0, finish_reason=r.choices[0].finish_reason)


class AnthropicClient(LLMClient):
    backend = "anthropic"

    def __init__(self, *a, api_key_env: str = "ANTHROPIC_API_KEY", **kw):
        super().__init__(*a, **kw)
        import anthropic
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s, max_retries=0)

    def _call(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        t0 = time.time()
        kw = {"system": system} if system else {}
        r = self._client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                         temperature=self.temperature, messages=msgs, **kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text=text, model=r.model, input_tokens=r.usage.input_tokens,
                           output_tokens=r.usage.output_tokens, latency_s=time.time() - t0,
                           finish_reason=r.stop_reason)


class MockClient(LLMClient):
    """Deterministic fake model. FOR UNIT TESTS ONLY - outputs are not model results."""
    backend = "mock"

    def _call(self, messages):
        last = messages[-1]["content"]
        h = int(hashlib.md5(last.encode()).hexdigest(), 16)
        if "Final answer: <number>" in last or "<number>" in last:
            ans = str(h % 50)
        else:
            ans = "ABCD"[h % 4]
        return LLMResponse(text=f"[MOCK] reasoning...\nFinal answer: {ans}", model="mock",
                           input_tokens=0, output_tokens=0, latency_s=0.0, finish_reason="stop")


BACKENDS = {"ollama": OllamaClient, "openai": OpenAIClient, "openai_compatible": OpenAIClient,
            "anthropic": AnthropicClient, "mock": MockClient}


def make_client(model_cfg: dict, gen_cfg: dict, seed: int, cache: ResponseCache | None) -> LLMClient:
    cfg = dict(model_cfg)
    backend = cfg.pop("backend")
    model = cfg.pop("model")
    kw = {**gen_cfg, **cfg}
    num_ctx = kw.pop("num_ctx", None)
    if backend == "ollama" and num_ctx:
        kw["num_ctx"] = num_ctx
    return BACKENDS[backend](model, seed=seed, cache=cache, **kw)
