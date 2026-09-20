from __future__ import annotations

import unittest

from docuocr.contract import ExtractionEnvelope


class ContractTests(unittest.TestCase):
    def test_exact_top_level_and_nested_aliases(self) -> None:
        envelope = ExtractionEnvelope.model_validate(
            {
                "data": {
                    "baby": {
                        "birthTime": "08:15",
                        "sample": {"collection": {"date": "2026-09-20"}},
                    },
                    "babyConsentforAllAndNgsTracking": True,
                    "payer": {"healthInsurance": "Example Health"},
                    "serialNumber": "SN-1",
                },
                "meta": {"job": {"id": "job-1"}},
            }
        ).as_public_dict()
        self.assertEqual(set(envelope), {"data", "meta"})
        self.assertEqual(envelope["data"]["baby"]["birthTime"], "08:15")
        self.assertIsInstance(envelope["data"]["baby"]["sample"], dict)
        self.assertTrue(envelope["data"]["babyConsentforAllAndNgsTracking"])
        self.assertEqual(envelope["meta"]["job"]["id"], "job-1")
        self.assertIn("createDocumentBlueprint", envelope["meta"]["timing"])

    def test_unknown_contract_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExtractionEnvelope.model_validate(
                {"data": {"unexpected": "unsafe"}, "meta": {"job": {"id": "job-1"}}}
            )


if __name__ == "__main__":
    unittest.main()
