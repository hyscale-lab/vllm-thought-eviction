# External Integrations

**Analysis Date:** 2026-04-07

## APIs & External Services

**OpenAI-Compatible API (served, not consumed):**
- vLLM exposes an OpenAI-compatible REST API for chat completions, completions, and responses
  - Implementation: `vllm/entrypoints/openai/`
  - Extended with `eviction_params` field on `ChatCompletionRequest` in `vllm/entrypoints/openai/chat_completion/protocol.py`

**Anthropic SDK (dependency):**
- `anthropic>=0.71.0` listed in `requirements/common.txt`
- Used for Anthropic-compatible entrypoint: `vllm/entrypoints/anthropic/`

**OpenAI SDK (dependency):**
- `openai>=1.99.1` listed in `requirements/common.txt`
- Used for client calls in benchmarks and the Responses API reasoning content path

**openai-harmony:**
- `openai-harmony>=0.0.3` listed in `requirements/common.txt`
- Used for GPT-OSS compatibility in `vllm/entrypoints/openai/chat_completion/stream_harmony.py`

**MCP (Model Context Protocol):**
- `mcp` package listed in `requirements/common.txt`
- Entrypoint at `vllm/entrypoints/mcp/`

## Model & Weight Sources

**HuggingFace Hub:**
- Used to download model weights and LoRA adapters
- Client: `huggingface_hub` package
- Usage: `vllm/lora/utils.py`, `vllm/assets/video.py`, `vllm/tokenizers/grok2.py`, `vllm/engine/arg_utils.py`
- Auth: `HUGGING_FACE_HUB_TOKEN` environment variable (standard HF convention)

**HuggingFace Transformers:**
- `transformers>=4.56.0,<5` — model config loading, base tokenizer
- Implementation: `vllm/transformers_utils/`

**Mistral Common:**
- `mistral_common[image]>=1.8.8` — Mistral/Tekken tokenizer, instruct protocol
- Implementation: `vllm/tokenizers/mistral.py`, `vllm/reasoning/mistral_reasoning_parser.py`

## Data Storage

**Databases:**
- None. vLLM is a stateless inference server; no database dependency.

**KV Cache (in-process):**
- GPU HBM for active KV cache blocks (managed by `vllm/v1/core/sched/scheduler.py`)
- CPU RAM for offloaded KV blocks (`vllm/v1/kv_offload/`)
- Eviction mask state stored in `scheduler.request_eviction_data: dict[str, list[tuple[int,int]]]`

**L2 Norm Cache (in-process singleton):**
- `L2NormCache` singleton in `vllm/v1/attention/l2_norm_cache.py`
- Pre-allocates 120 KB CPU `torch.Tensor` buffer per active eviction request
- Thread-safe via `threading.Lock` per `RequestL2NormData`

**File Storage:**
- LoRA adapter weights loaded from local filesystem or HuggingFace Hub
- LoRA filesystem resolver plugin: `vllm/plugins/lora_resolvers/filesystem_resolver.py`
- Structured output disk cache: `diskcache==5.6.3`

**Caching:**
- `diskcache==5.6.3` — disk-backed cache for outlines backend schemas
- `cachetools` — in-process LRU/TTL caches throughout vllm

## Authentication & Identity

**Auth Provider:**
- None built-in. API key validation (if any) is handled upstream (e.g., reverse proxy).
- HuggingFace Hub token via `HUGGING_FACE_HUB_TOKEN` env var for model downloads.

## Monitoring & Observability

**Metrics:**
- `prometheus_client>=0.18.0` — exposes `/metrics` endpoint
- `prometheus-fastapi-instrumentator>=7.0.0` — HTTP request/latency metrics
- Implementation: `vllm/v1/metrics/prometheus.py`, `vllm/v1/metrics/loggers.py`, `vllm/v1/metrics/reader.py`

**Error Tracking:**
- None (no Sentry or similar). Errors are logged via Python's standard `logging` module via `vllm/logger.py` (`init_logger()`).

**Logs:**
- Python logging throughout; `python-json-logger` available for structured JSON log output
- `setproctitle` — sets process names for worker visibility in `ps` / monitoring tools

**Profiling:**
- `depyf==0.20.0` — profiling/debugging with torch.compile; `vllm/profiler/`

## IPC & Distributed Communication

**ZeroMQ:**
- `pyzmq>=25.0.0` — engine core ↔ API server IPC
- Patterns used: DEALER/XSUB/PUSH sockets in `vllm/v1/engine/core.py` and `vllm/v1/engine/core_client.py`
- `update_request_mask` eviction commands and L2 norm data travel over this channel

**Ray:**
- `ray[cgraph]>=2.48.0` — multi-GPU pipeline parallelism, worker placement
- Used in: `vllm/v1/engine/utils.py`, `vllm/ray/`
- `cgraph` (compiled graph) required for pipeline parallelism in V1 engine

**gRPC:**
- `grpcio>=1.76.0` — alternative engine serving protocol
- Proto definition: `vllm/grpc/vllm_engine.proto`
- Generated stubs: `vllm/grpc/vllm_engine_pb2.py`, `vllm/grpc/vllm_engine_pb2_grpc.py`
- Server entrypoint: `vllm/entrypoints/grpc_server.py`
- gRPC reflection enabled via `grpcio-reflection>=1.76.0`

**msgspec / msgpack:**
- Zero-copy binary serialization for IPC messages between engine core and detokenizer/output processor
- Used in: `vllm/v1/serial_utils.py`, `vllm/v1/engine/coordinator.py`, `vllm/v1/engine/core_client.py`

## CI/CD & Deployment

**Hosting:**
- Docker images provided for CUDA, ROCm, CPU, TPU, XPU targets (`docker/`)
- Docker Bake configuration: `docker/docker-bake.hcl`

**CI Pipeline:**
- Buildkite: `.buildkite/test-pipeline.yaml`, `.buildkite/test-amd.yaml`, `.buildkite/release-pipeline.yaml`
- Code coverage: `codecov.yml`

## Webhooks & Callbacks

**Incoming:**
- None. vLLM is a request/response and SSE streaming server only.

**Outgoing:**
- None in core. Benchmarks (`benchmarks/benchmark_serving.py`) make outbound HTTP requests to a running vLLM instance for load testing purposes.

## Structured Output Backends

**xgrammar:**
- `xgrammar==0.1.29` — grammar-constrained decoding; `vllm/v1/structured_output/`
- Available on x86_64, aarch64, arm64, s390x, ppc64le

**Outlines Core:**
- `outlines_core==0.2.11` — FSM-based constrained decoding

**lm-format-enforcer:**
- `lm-format-enforcer==0.11.3` — format-constrained generation

**llguidance:**
- `llguidance>=1.3.0,<1.4.0` — guidance-based backend
- Implementation: `vllm/v1/structured_output/backend_guidance.py`

## Environment Configuration

**Key env vars (noted, contents not read):**
- `VLLM_TARGET_DEVICE` — `cuda` / `rocm` / `cpu` / `tpu` / `xpu`; build and runtime target
- `HUGGING_FACE_HUB_TOKEN` — model download authentication
- `VLLM_DISABLE_SCCACHE` — disable sccache compiler cache
- `CUDA_HOME` — path to CUDA toolkit for native extension compilation
- `ROCM_HOME` — path to ROCm for HIP builds

---

*Integration audit: 2026-04-07*
