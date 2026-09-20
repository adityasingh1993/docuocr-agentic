# Architecture research notes

The design was checked against current primary documentation on 20 September 2026.

## Orchestration

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): `StateGraph`, typed shared state, nodes, normal edges, conditional edges, reducers, and compilation.
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence): thread-scoped checkpoints; SQLite is positioned for local workflows and PostgreSQL for production.
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts): durable pause/resume for human review when compiled with a checkpointer and invoked with a thread ID.

Those primitives directly map to this repository's typed document state, parallel evidence fan-out, confidence routing, bounded recovery loop, SQLite developer profile, PostgreSQL production profile, and optional reviewer interrupt.

## OCR and document parsing

- [PaddleOCR-VL pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html): two-stage layout detection and vision-language recognition; parsing blocks expose IDs/order, labels, content, and bounding boxes.
- [PaddleOCR general OCR pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html): box-level recognized text and recognition scores, with configurable detection/box/recognition thresholds.
- [PaddleOCR-VL on Apple Silicon](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html): supported Apple Silicon path and MLX-VLM service integration.
- [PaddleOCR-VL on NVIDIA Blackwell](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html): Blackwell-specific Paddle/CUDA installation requirements and deployment modes.

PaddleOCR-VL is therefore used for structure, while general PaddleOCR remains the exhaustive text ledger. OpenCV independently handles image quality and form controls.

## Local VLM

- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct): Apache-2.0 4B vision-language model with OCR/document capabilities.
- [Official Qwen3-VL-4B-Instruct GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF): official quantized files, including a roughly 2.5 GB Q4_K_M language model plus vision projector, and a `llama-server` example.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): local OpenAI-compatible server with Metal and CUDA backends and quantization support.
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm): Apple Silicon VLM inference/server with OpenAI-compatible endpoints and structured output support.

Qwen3-VL 4B Q4 is the portability default. It should run with concurrency one, bounded image dimensions, and bounded output tokens on an 8 GB GPU. The exact usable context/cache depends on backend and image resolution and must be load-tested on the target machine.

## Safety conclusion

Model-reported confidence is not used. The system accepts values only through observable evidence, deterministic validation, independent agreement, and a locally calibrated policy. The shipped `0.90` threshold is a workflow policy until field-specific calibration is fitted on labelled production-like documents.
