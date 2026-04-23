# Technology Stack

**Analysis Date:** 2026-04-07

## Languages

**Primary:**
- Python 3.12 (runtime on this machine; supported range 3.10–3.13) - All vLLM engine, serving, and thought eviction logic
- C++17 - CUDA/HIP extension kernels in `csrc/`
- CUDA C (.cu) - GPU attention, cache, norm, and sampling kernels in `csrc/`

**Secondary:**
- CMake - Build system for C++/CUDA extensions (`CMakeLists.txt`, `cmake/`)
- Protobuf / gRPC IDL - Engine IPC definition at `vllm/grpc/vllm_engine.proto`
- Jinja2 - Chat template rendering (`examples/template_*.jinja`)

## Runtime

**Environment:**
- CPython 3.10–3.13 (3.12 installed on this machine)
- Linux (primary) or macOS (CPU-only fallback); WSL supported

**Package Manager:**
- pip / uv (uv config present in `pyproject.toml` `[tool.uv]`)
- No lockfile committed; `requirements/*.txt` pin versions

## Frameworks

**Core Serving:**
- FastAPI >= 0.115.0 - OpenAI-compatible HTTP API server (`vllm/entrypoints/openai/`)
- Uvicorn - ASGI server, launched in `vllm/entrypoints/launcher.py`

**ML / Computation:**
- PyTorch 2.9.1 - Tensor ops, CUDA integration, model execution (all of `vllm/model_executor/`, `vllm/v1/`)
- FlashInfer-Python 0.5.3 - Paged attention kernels (`vllm/v1/attention/backends/flashinfer.py`)
- Numba 0.61.2 - N-gram speculative decoding (`requirements/cuda.txt`)
- NumPy - L2 norm arrays in thought eviction (`vllm/thought_eviction/strategies.py`, `vllm/thought_eviction/orchestrator.py`, `vllm/v1/attention/l2_norm_cache.py`)
- Triton (bundled with PyTorch + vendored in `vllm/third_party/triton_kernels/`) - Custom GPU kernels for MoE, topk, matmul

**Distributed Execution:**
- Ray >= 2.48.0 with `ray[cgraph]` - Pipeline parallelism, multi-node worker lifecycle (`vllm/v1/engine/utils.py`, `vllm/ray/`)
- ZeroMQ (pyzmq >= 25.0.0) - Inter-process communication between engine core and API server (`vllm/v1/engine/core.py`, `vllm/v1/engine/core_client.py`)
- msgspec + msgpack - Zero-copy IPC serialization (`vllm/v1/serial_utils.py`, `vllm/v1/engine/coordinator.py`)

**Tokenization:**
- HuggingFace Transformers >= 4.56.0 - Model config, tokenizer base (`vllm/transformers_utils/`)
- HuggingFace tokenizers >= 0.21.1 - Fast incremental detokenization
- sentencepiece - LLaMA tokenizer
- tiktoken >= 0.6.0 - DBRX tokenizer (`vllm/tokenizers/grok2.py`)
- mistral_common >= 1.8.8 - Mistral/Tekken tokenizer and instruct protocol (`vllm/tokenizers/mistral.py`)

**Schema / Validation:**
- Pydantic >= 2.12.0 - Request/response models throughout entrypoints and thought eviction params (`vllm/entrypoints/openai/chat_completion/protocol.py`)

**gRPC:**
- grpcio >= 1.76.0 - Engine gRPC server (`vllm/entrypoints/grpc_server.py`)
- grpcio-tools >= 1.76.0 - Proto compilation at build time (`setup.py`)
- grpcio-reflection >= 1.76.0 - gRPC server reflection
- MCP (model context protocol) - `vllm/entrypoints/mcp/`

**Structured Output:**
- xgrammar 0.1.29 - Grammar-constrained decoding
- outlines_core 0.2.11 - Outlines-based structured output
- lm-format-enforcer 0.11.3 - Format-enforced generation
- llguidance >= 1.3.0 - Guidance-based backend (`vllm/v1/structured_output/backend_guidance.py`)

**Testing:**
- pytest - Test runner (configured in `pyproject.toml` `[tool.pytest.ini_options]`)

**Linting / Formatting:**
- Ruff - Linting and formatting (configured in `pyproject.toml` `[tool.ruff]`)
- mypy - Static type checking (configured in `pyproject.toml` `[tool.mypy]`)
- pre-commit 4.0.1 - Git hooks (`requirements/lint.txt`, `.pre-commit-config.yaml`)
- typos - Spell checker (configured in `pyproject.toml` `[tool.typos]`)

## Key Dependencies

**Critical:**
- `torch==2.9.1` - All GPU compute; pinned exactly (`requirements/cuda.txt`, `pyproject.toml`)
- `flashinfer-python==0.5.3` - Paged attention for CUDA path; must match torch version
- `ray[cgraph]>=2.48.0` - Required for pipeline parallelism
- `pyzmq>=25.0.0` - Engine IPC backbone
- `transformers>=4.56.0,<5` - Model loading and tokenization

**Thought Eviction Specific:**
- `numpy` - L2 norm array operations in `vllm/thought_eviction/strategies.py` and `vllm/v1/attention/l2_norm_cache.py`
- `torch` (CPU tensors) - `RequestL2NormData` pre-allocates a 120KB CPU `torch.zeros` buffer per request in `vllm/v1/attention/l2_norm_cache.py`

**Infrastructure:**
- `prometheus_client>=0.18.0` - Metrics (`vllm/v1/metrics/`)
- `prometheus-fastapi-instrumentator>=7.0.0` - HTTP server metrics
- `pydantic>=2.12.0` - All API models including `EvictionParams` in `vllm/entrypoints/openai/chat_completion/protocol.py`
- `msgspec` - Fast IPC serialization for scheduler ↔ engine boundary
- `fastapi[standard]>=0.115.0` - HTTP serving layer

## Configuration

**Environment:**
- `VLLM_TARGET_DEVICE` - Controls build target: `cuda`, `rocm`, `cpu`, `tpu`, `xpu`; auto-detected from torch in `setup.py`
- `VLLM_DISABLE_SCCACHE` - Disable sccache for compilation
- Environment variables loaded from `vllm/envs.py` at build time

**Build:**
- `pyproject.toml` - Project metadata, build backend, ruff/mypy/pytest config
- `CMakeLists.txt` - C++/CUDA extension build (requires cmake >= 3.26.1)
- `cmake/cpu_extension.cmake`, `cmake/utils.cmake` - Platform-specific build helpers
- `requirements/build.txt` - Build-time Python dependencies (cmake, ninja, torch, jinja2, grpcio-tools)
- `setup.py` - Custom build logic; compiles gRPC protos, dispatches to CMake

## Platform Requirements

**Development:**
- Linux or macOS (Linux required for CUDA/ROCm)
- CUDA toolkit (for CUDA target) - `CUDA_HOME` must be set; supported archs 7.0–12.0 depending on nvcc version
- ROCm (for ROCm target) - `ROCM_HOME`; `cmake/hipify.py` converts CUDA sources
- Ninja build tool (recommended)

**Production:**
- NVIDIA GPU (CUDA 12.x or 13.x recommended for sm_90+ / Hopper)
- AMD GPU (ROCm path via `csrc/rocm/`, `requirements/rocm.txt`)
- Google TPU (optional, `requirements/tpu.txt`)
- Intel GPU/XPU (optional, `requirements/xpu.txt`)
- Deployable via Docker (`docker/Dockerfile`, `docker/Dockerfile.rocm`, etc.)

---

*Stack analysis: 2026-04-07*
