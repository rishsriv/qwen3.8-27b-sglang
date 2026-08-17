# Qwen3.8-27B NVFP4 on SGLang

This repository runs [`RadixArk/Qwen3.8-27B-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) on a single NVIDIA RTX 5090 with [SGLang](https://github.com/sgl-project/sglang). The deployment favors context capacity over speculative-decoding throughput.

The service exposes SGLang's OpenAI-compatible API, including `/v1/responses` for Codex, at:

```text
http://127.0.0.1:30000/v1
```

The server binds only to loopback and every API request requires a bearer token. The token is intentionally not committed to this repository. The launcher reads it at startup, removes it from the worker environment, and injects it into SGLang internally rather than putting it in the process command line.

The local deployment was verified end to end on August 18, 2026:

- an anonymous local request to `/v1/models` returned `401`
- authenticated local `/v1/models` and `/v1/responses` requests returned `200`
- a local Responses API generation reached `completed`
- Codex CLI 0.147.0 connected through the custom provider with `medium` reasoning effort
- a Codex function call was returned in the expected Responses API shape

## Host setup

Prerequisites:

- Linux with an NVIDIA Blackwell GPU and a recent driver
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- FFmpeg for the checkpoint's optional audio/video preprocessing
- Tailscale, if the public HTTPS endpoint is wanted

On Ubuntu, install FFmpeg with:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Install the pinned SGLang release in an isolated environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --prerelease=allow -r requirements.txt
```

The CUDA compiler, NVVM frontend, and CRT pins in `requirements.txt` are
intentional. Unconstrained prerelease resolution can mix CUDA 13.4 compiler
components with the CUDA 13.0 headers installed by PyTorch.

Install and start the user service:

```bash
chmod +x scripts/*.sh
./scripts/install-service.sh
journalctl --user -u qwen3.8-sglang.service -f
```

The first start downloads the Hugging Face checkpoint. The API key is generated once at:

```text
~/.local/state/qwen3.8-sglang/api-key
```

Optional: expose the loopback-only server through persistent HTTPS when remote access is needed. The live deployment currently leaves Funnel disabled:

```bash
tailscale funnel --bg --yes 30000
tailscale funnel status
```

Check the local API after startup:

```bash
./scripts/check.sh
```

When Funnel is enabled, check the public path from a machine outside this host's tailnet:

```bash
SGLANG_BASE_URL=https://rishabh-rtx5090.taild7d3df.ts.net/v1 ./scripts/check.sh
```

On the server itself, use the default loopback check. Tailscale MagicDNS may
resolve the public hostname to the host's private tailnet address instead of a
Funnel relay, so an on-host request does not reliably exercise the public path.

## Use it from Codex

Codex custom providers use the Responses API. Copy or merge
[`codex-config.toml`](./codex-config.toml) into `~/.codex/config.toml`; it
selects this model and makes `medium` the default reasoning effort. Keep
[`codex-model-catalog.json`](./codex-model-catalog.json) beside this clone and
update the absolute `model_catalog_json` path in the TOML if the clone lives
somewhere other than `/home/rishabh-srivastava/qwen`.

Put the server token in the shell environment, then start Codex:

```bash
export SGLANG_API_KEY='token-provided-by-the-server-owner'
codex
```

The important Codex settings are:

```toml
model = "qwen3.8-27b-nvfp4"
model_provider = "qwen38_sglang"
model_reasoning_effort = "medium"
model_supports_reasoning_summaries = false
model_catalog_json = "/absolute/path/to/qwen3.8-27b-sglang/codex-model-catalog.json"

[model_providers.qwen38_sglang]
name = "Qwen3.8 27B NVFP4 (SGLang)"
base_url = "http://127.0.0.1:30000/v1"
env_key = "SGLANG_API_KEY"
wire_api = "responses"
```

The checkpoint advertises a 262,144-token context, while the service's
practical token pool is about 200,000 tokens on a single 32 GB RTX 5090. Set
the catalog's truncation and automatic-compaction limits conservatively below
that boundary to retain headroom for generated tokens and runtime variation.

On the server owner account, load the generated token without printing it:

```bash
export SGLANG_API_KEY="$(<~/.local/state/qwen3.8-sglang/api-key)"
codex
```

Codex sends tools to the model through SGLang's direct `/v1/responses` implementation. The deployment deliberately does not place SGLang Model Gateway in front of the server, avoiding an extra tool-schema compatibility layer.

## Runtime compatibility notes

SGLang 0.5.17 is pinned because this deployment was tested end to end on the
RTX 5090. The launch wrapper also handles three current wheel/runtime details:

- conventional CUDA `lib64` linker names for the wheel-provided runtime and cuBLAS
- `MAX_JOBS=4` so first-start FlashInfer compilation fits in 61 GiB host RAM
- a narrow compatibility shim that maps Codex's `high` reasoning effort to the
  checkpoint template's equivalent `xhigh` tier during prompt rendering

The first successful boot can take several minutes while FlashInfer compiles
and caches Blackwell kernels. Later restarts reuse that cache and are much
faster.

## Deployment choices

The launch flags follow SGLang's verified RTX 5090 recipe:

- NVFP4 target weights and FP8 KV cache
- FlashInfer attention without a speculative draft model
- BF16 Mamba state to maximize the practical token pool
- Qwen reasoning and `qwen3_coder` tool parsers
- one concurrent request and decode CUDA graph batch size 1
- five persistent Mamba state-cache slots for the single-request service
- API-key authentication and loopback-only binding

The 32 GB RTX 5090 configuration is optimized for one interactive coding-agent session. Increasing concurrency requires re-deriving the Mamba cache and static-memory settings; simply raising `--max-running-requests` can exhaust the GDN state pool.

## Operations

```bash
systemctl --user status qwen3.8-sglang.service
journalctl --user -u qwen3.8-sglang.service -n 200
systemctl --user restart qwen3.8-sglang.service
tailscale funnel status
nvidia-smi
```

The checkpoint has a native context window of 262,144 tokens, but the practically available context depends on runtime memory use and prompt shape. Validate your real Codex workload before relying on the maximum.
