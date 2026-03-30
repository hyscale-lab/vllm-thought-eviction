# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/contributing/benchmarks.html#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

## Request composition for thought-eviction benchmarks

For benchmark clients that call this server directly, compose requests as
OpenAI-compatible chat completions with the eviction extension fields below.

### 1) Primary generation request

- Endpoint: `POST /v1/chat/completions`
- Body: standard OpenAI chat body (`model`, `messages`, `stream`, etc.)
- Eviction config must be sent in **`eviction_params`** (not `eviction`).

Minimal example:

```json
{
  "model": "your-model",
  "messages": [{"role": "user", "content": "Explain ..."}],
  "stream": true,
  "request_id": "bench-run-001",
  "eviction_params": {
    "strategy": "thought_min",
    "keep_ratio": 0.7,
    "trigger_mode": "time",
    "eviction_interval_tokens": 256,
    "retention_window_tokens": 512,
    "prune_after_tokens": 512,
    "min_segment_tokens": 15,
    "protect_first_thought": true,
    "l2_norm_layers": [8, 10]
  }
}
```

Notes:

- `strategy` should be set when `eviction_params` is present (`global`,
  `thought_min`, `thought_avg`, or `random`).
- `l2_norm_layers` is now per-request and belongs inside `eviction_params`.
- `request_id` is optional, but recommended for benchmarking traceability.
  The server response id is returned as `chatcmpl-<request_id>` (or from
  `X-Request-Id` header if provided).

### 2) L2 norm polling request (optional)

To fetch accumulated norms while a request is running:

- Endpoint: `POST /v1/attention/l2_norms`
- Body:

```json
{
  "request_id": "chatcmpl-bench-run-001",
  "start_index": 0
}
```

Use the `id` from the chat completion stream/response for `request_id`, and
increase `start_index` as you consume results.