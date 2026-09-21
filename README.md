# DocuOCR Agentic

DocuOCR Agentic is a private, local-first form extraction starter. It keeps recognition, computer vision, semantic mapping, confidence policy, and the public JSON contract separate so each part can be changed or scaled independently.

The repository is deliberately **bounded-agentic**. LangGraph can retry approved transformations and ask local models for grounded proposals, but a model cannot invent tools, change the schema, accept an ungrounded value, or loop indefinitely.

## Processing contract

1. Ingest the image, hash it, and create an audit run.
2. Score resolution, blur, contrast, illumination, glare, and skew without changing the original image.
3. In parallel, acquire baseline evidence:
   - parse layout with PaddleOCR-VL;
   - run exhaustive box-level PaddleOCR;
   - detect checkbox/radio controls with OpenCV.
4. Ask Qwen3-VL once for bounded document understanding: form/card family match, languages, regions, control groups, quality issues, and handwriting legibility. It never returns field values in this pass.
5. When readiness is low, generate up to three approved OpenCV variants. Keep the original unless a fresh OCR pass shows a configured gain while retaining baseline text and controls; then reacquire evidence from the selected image.
6. Resolve the configured form blueprint and generate geometry-based field candidates. An explicit VLM blueprint mismatch forces review rather than extraction against the wrong schema.
7. Ask the local VLM for unresolved mappings and visual re-checking of every checkbox/radio group. The VLM must cite existing OCR plus actual-image evidence IDs.
8. Normalize and validate values with Pydantic and cross-field rules.
9. Score every field. Values below `0.90` receive targeted crop enhancement and independent re-verification.
10. Accept a value only when it is grounded, valid, and above policy. Otherwise pause/queue it for review.
11. Emit the requested `data`/`meta` JSON plus a local evidence and trace bundle.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design and scaling path.

## Why three vision paths

| Path | Responsibility | Never trusted for |
| --- | --- | --- |
| PaddleOCR | exhaustive text, scores, polygons | semantic field meaning by itself |
| OpenCV | technical quality, approved image variants, checkbox/radio state | reading names or dates |
| Local VLM | document structure, handwriting legibility, unfamiliar layout interpretation, crop verification | unsupported values, free-form image editing, or self-confidence |

PaddleOCR-VL remains useful for page structure, but its Markdown is not treated as the complete transcription. The general OCR pass is the evidence ledger.

## Local VLM profiles

The application speaks to an OpenAI-compatible endpoint through `LocalVLMClient`. External hosts are rejected by default.

VLM requests are split into small field batches. If a local server returns a
truncated or malformed JSON batch, DocuOCR splits the batch and retries smaller
requests. A malformed single-field response is recorded as a warning instead of
raising `JSONDecodeError` and terminating the document job.

### Portable Mac or NVIDIA profile: llama.cpp

Use the official Qwen GGUF and a local server:

```bash
llama serve \
  -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M \
  --host 127.0.0.1 \
  --port 8081 \
  --ctx-size 4096 \
  --parallel 1
```

The Q4 language weights are about 2.5 GB; the vision projection and runtime cache also consume memory. On an 8 GB GPU, keep concurrency at one, restrict image resolution/token output, and put Paddle/VLM processes behind one GPU-admission queue. If the combined resident set is too large, run general OCR/layout on CPU and reserve the GPU for crop-level VLM verification.

### Apple Silicon profile: MLX-VLM

```bash
python -m pip install "mlx-vlm>=0.3.11"

mlx_vlm.server \
  --model mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8081
```

For this profile, change `vlm.model` in `config/settings.yaml` to
`mlx-community/Qwen3-VL-4B-Instruct-4bit`. Keep the configured model ID aligned
with the server's loaded model.

When Windows calls an MLX-VLM server running on another private-network machine,
enable the private endpoint explicitly:

```yaml
vlm:
  enabled: true
  base_url: http://<mac-ip>:8082/v1
  model: mlx-community/Qwen3-VL-4B-Instruct-4bit
  api_key: local-only
  max_tokens: 2048
  document_understanding_enabled: true
  understanding_max_tokens: 1024
  max_understanding_ocr_spans: 80
  max_paths_per_request: 4
  max_controls_per_request: 32
  request_retries: 1
  allow_private_lan: true
```

`api_key` is an arbitrary shared placeholder unless the local server has been
configured to validate a real bearer token.

For PaddleOCR-VL itself, Apple Silicon can use its PaddlePaddle layout stage and an MLX-VLM recognition service. Blackwell GPUs require the Blackwell-specific Paddle/CUDA build described in the official PaddleOCR guide.

## Installation

Create a clean environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the core project:

```bash
python -m pip install -e ".[test]"
```

For real-image OCR with Paddle-format models, install the PaddlePaddle backend:

```bash
python -m pip install -e ".[paddle]"
```

Install a different backend only when the local model directories use that format:

```bash
# Hugging Face / safetensors models
python -m pip install -e ".[paddle-transformers]"

# ONNX models
python -m pip install -e ".[paddle-onnx]"
```

PaddleOCR recommends one inference backend per environment. The selected backend and
model files must match: Paddle inference models use `engine: paddle`, while Hugging
Face/safetensors models use `engine: transformers`.

### Apple Silicon

Use a native ARM64 Python environment on M-series Macs. A virtual environment created
from an Intel Homebrew Python under `/usr/local` remains `x86_64` and can force Rust
packages such as `cryptography` to build from source.

```bash
arch -arm64 /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv-arm64
source .venv-arm64/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -c "import platform; print(platform.machine())"  # must print arm64
python -m pip install --only-binary=:all: cryptography
python -m pip install -e ".[paddle]"
```

The Python.org universal2 build is also usable by invoking it with `arch -arm64`.
If `--only-binary` cannot find `cryptography`, the configured package mirror must add
the macOS ARM64 wheel; do not silently fall back to an Intel source build.

## Configure local model paths

Copy and edit `config/settings.yaml`. In a private or offline deployment, every Paddle model directory must be explicit and already present locally. `offline: true` prevents an accidental model-host lookup.

Start with real-image text OCR only. This does not use a sidecar and avoids loading the
larger layout/VL pipeline until its matching model files have been verified:

```yaml
paddle:
  enabled: true
  text_enabled: true
  layout_enabled: false
  engine: paddle
  device: cpu
  text_detection_model_dir: /absolute/path/to/PP-OCRv6_medium_det
  text_recognition_model_dir: /absolute/path/to/PP-OCRv6_medium_rec

vlm:
  enabled: false
```

`text_engine` and `layout_engine` may override `engine` when the two stages are
deployed separately. Enable `layout_enabled` only after `layout_model_dir` and
`vl_rec_model_dir` have been supplied in the selected layout engine's format.

The example newborn-screening blueprint is in `config/blueprints/newborn_screening.yaml`. Add another YAML blueprint for each form family or country variant; do not fork the extraction code.

The document-understanding pass compares the image with the configured blueprint and
can return `match`, `possible`, `mismatch`, or `unknown`. The current CLI still receives
one configured blueprint; it does not silently choose an unrestricted schema. A
`mismatch` routes all fields to review. A future registry can use the same bounded
analysis to select among an explicit set of known blueprints.

Blueprint aliases may contain multiple languages. Deterministic rules and the
VLM use translated/canonical printed labels only to locate a field. Entered or
handwritten names, identifiers, dates, and free text are never translated; their
raw OCR text remains the evidence. For a control such as `M`/`F`, the characters
are option labels. Selection comes from visible ink—circle, tick, cross, or fill—
in the actual image, not from OCR merely recognizing the option text.

## Run

With installed local backends, pass the actual image as the positional argument and
do not include `--sidecar`:

```bash
docuocr extract path/to/card.jpg \
  --config config/settings.yaml \
  --blueprint config/blueprints/newborn_screening.yaml
```

Validate the output contract or print its JSON Schema:

```bash
docuocr schema
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

To exercise orchestration before installing Paddle, pass an evidence sidecar. This is
also the recommended contract test between a separately deployed OCR service and the
workflow:

```bash
docuocr extract path/to/card.jpg \
  --config config/settings.yaml \
  --blueprint config/blueprints/newborn_screening.yaml \
  --sidecar examples/demo_sidecar.json \
  --job-id demo-001
```

Sidecar mode is deterministic test plumbing, not a production recognizer.

## Output and audit files

Each run writes under `artifacts/<job-id>/`:

```text
result.json
trace.jsonl
evidence.json
images/
  original-reference.json
  document-candidate-*.png
  crop-*.png
document-understanding.json  # when the VLM analysis succeeds
```

The trace stores hashes, model/config identities, transformation parameters, routing decisions, and evidence references. Raw field values are omitted from trace records by default. The protected `evidence.json` ledger stores OCR text, boxes, controls, quality history, document understanding, enhancement evaluations, and hashes so each decision can be audited; treat it as sensitive data. The business output keeps normalized values in `data` and raw pre-normalization values in `meta.pre`.

Full-page enhancement is deliberately non-generative. Qwen can recommend only
`deskew`, `local_contrast`, `illumination_normalization`,
`mild_denoise_sharpen`, `upscale`, or `none`. OpenCV uses fixed parameters, and
full-page adaptive thresholding is prohibited because it can erase thin handwriting
or create false checkbox marks.

## Confidence warning

`0.90` is a policy threshold, not automatically a 90% probability. OCR/VLM scores are not interchangeable. The included scorer is intentionally conservative and marks itself uncalibrated. Before production auto-acceptance, fit an isotonic or logistic calibrator on a held-out, manually labelled set and measure false auto-acceptance per critical field and script.

## Production scaling

- Run the API and LangGraph workers separately.
- Use PostgreSQL checkpointers instead of SQLite.
- Put encrypted evidence artifacts in an on-prem object store.
- Keep PaddleOCR and VLM services behind private network endpoints.
- Route by model capability and GPU memory; use one VLM request at a time on 8 GB cards.
- Send only low-confidence crops to the VLM, not every full page.
- Export OpenTelemetry traces to a local collector; never send document content to hosted tracing.

## Privacy defaults

- No external model or telemetry endpoint is allowed by default.
- Model prompts and trace logs contain evidence IDs and hashes, not patient values, unless explicitly enabled.
- Original and enhanced images require a local retention policy and encryption at rest.
- Human review is part of the correctness boundary, not an operational failure.
