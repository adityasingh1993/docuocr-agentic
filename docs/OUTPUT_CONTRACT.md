# Output contract

The public object contains exactly `data` and `meta`. All fields are present through Pydantic defaults even when a value is `null`; nested hierarchy objects such as `baby.sample.collection` remain objects.

```json
{
  "data": {
    "baby": {
      "birthTime": "08:15",
      "birthWeight": "3.25 kg",
      "dateOfBirth": "2024-08-12",
      "female": true,
      "firstName": "Amina",
      "lastName": "Example",
      "male": false,
      "medication": null,
      "nationality": null,
      "parenteral": null,
      "registerNumber": "REG-100",
      "sample": {
        "collection": {
          "date": "2024-08-13",
          "time": "09:10"
        }
      }
    },
    "babyConsentforAllAndNgsTracking": true,
    "babyFirstTimeScreening": true,
    "babyHearingScreenPerformed": false,
    "mother": {},
    "payer": {"healthInsurance": "Example Health"},
    "serialNumber": "SN-42"
  },
  "meta": {
    "identifiers": {"serialNumber": "SN-42"},
    "job": {"id": "job-42"},
    "pre": {
      "baby": {
        "birthTime": "8:15 AM",
        "dateOfBirth": "12/08/2024",
        "sample": {"collection": {"date": "13/08/2024", "time": "09:10"}}
      }
    },
    "processors": {
      "steps": [],
      "fields": {},
      "evidenceManifest": "artifacts/job-42/evidence.json",
      "layoutManifest": "artifacts/job-42/layout-extractions.json",
      "traceManifest": "artifacts/job-42/trace.jsonl"
    },
    "timing": {
      "createDocumentBlueprint": 1.2,
      "extraction": 812.4,
      "getExtractionPrompt": 0.8,
      "total": 812.4,
      "steps": {}
    }
  }
}
```

Timings are milliseconds. `meta.processors.fields` is keyed by canonical dotted path and records confidence, calibration status, disposition, attempts, evidence references, validation codes, and source. It is the decision/audit layer without changing the requested business-data hierarchy.

`meta.processors.layoutManifest` points to a protected audit file whose `layouts`
array records OCR spans, controls, candidates, retries, and warnings per layout
block. Its final `combined` object contains the assembled data, accepted and
unresolved paths, and field decisions. The file contains extracted values and must
be handled with the same controls as `evidence.json`.
