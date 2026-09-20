from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from docuocr.contract import ExtractionEnvelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docuocr")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="extract a form image")
    extract.add_argument("source")
    extract.add_argument("--config", required=True)
    extract.add_argument("--blueprint", required=True)
    extract.add_argument(
        "--sidecar", help="deterministic evidence JSON instead of Paddle engines"
    )
    extract.add_argument("--job-id")
    schema = commands.add_parser("schema", help="print the public JSON Schema")
    schema.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "schema":
        rendered = json.dumps(
            ExtractionEnvelope.model_json_schema(by_alias=True), indent=2
        )
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    from docuocr.pipeline import DocumentPipeline

    with DocumentPipeline.from_files(
        args.config, args.blueprint, sidecar_path=args.sidecar
    ) as pipeline:
        result = pipeline.extract(args.source, job_id=args.job_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
