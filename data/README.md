# Data preparation

No clinical records are committed to this repository. Put the following files
in this directory (or set `JCFDM_DATA_DIR` to another directory):

| File | Rows | Model inputs | Outcome columns | Access |
|---|---:|---:|---|---|
| `support.csv` | 8,873 | 14 | `duration`, `event` | public through pycox |
| `gbsg.csv` | 2,232 | 7 | `duration`, `event` | public Rotterdam--GBSG benchmark through pycox |
| `seer.csv` | 11,600 | 61 | `time`, `cod` | controlled/restricted |

Prepare the two public files from the repository root with:

```bash
python scripts/prepare_public_data.py
python scripts/validate_data.py
```

The downloader uses `pycox.datasets.support.read_df()` and
`pycox.datasets.gbsg.read_df()`. It contacts the public data host on first use.

The SEER cohort is the encoded localized-pancreatic-cancer cohort distributed
with the DCSurvival study: diagnoses from 2000--2015 and pancreatic-cancer
death as the event. It is governed by the NCI/SEER data-use terms and is not
redistributed here. Authorized users should place the exact preprocessed file
at `data/seer.csv`. The loader expects 63 columns total: `time`, `cod`, and 61
numeric model inputs.

Reference SHA-256 fingerprints of the files used for the paper are:

```text
17a7aca1b760004e991998bdf472eecadbfcad31902de1e4ee022b95b7d2b2d6  support.csv
d9b457518036da5acab0741f9bd6872f12e5a7a6b9c7c00860ffc8f779965d0b  gbsg.csv
c57e9841391c941549bdc43dc906ce63de25800c979a8096f87aa9a0000a6373  seer.csv
```

A public-data re-download can differ byte-for-byte because of CSV formatting;
the validation script therefore treats public-file hash mismatches as warnings
after checking the dimensions and required columns. The SEER fingerprint is
the strongest available check for the exact restricted cohort.
