from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    def test_paddle_extras_include_compatible_engines(self) -> None:
        payload = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        extras = payload["project"]["optional-dependencies"]
        self.assertIn("paddlepaddle>=3.3,<4", extras["paddle"])
        self.assertIn(
            "transformers>=5.10,<6", extras["paddle-transformers"]
        )
        for name in ("paddle", "paddle-transformers", "paddle-onnx"):
            self.assertIn("paddleocr[doc-parser]>=3.5,<4", extras[name])


if __name__ == "__main__":
    unittest.main()
