# Main-text reference values

These are the values printed in the paper. They are a
verification target, not cached replacements for running the experiments.

## Table 1: uninformative censoring

Columns are LogN-base, Heavy-tail-t, Cox-T/LogN-C, and Cox-both.

| Method | T-W1 | T-KS |
|---|---|---|
| Binned KM smoother | 0.386 / 1.181 / 1.786 / 0.191 | 0.126 / 0.176 / 0.364 / 0.051 |
| AFT-LogNormal | 0.070 / 0.580 / 1.213 / 0.320 | 0.039 / 0.107 / 0.193 / 0.065 |
| AFT-Weibull | 0.083 / 0.140 / 0.176 / 0.164 | 0.042 / 0.086 / 0.038 / 0.036 |
| T-only flow | 0.119 / 0.310 / 0.450 / 0.243 | 0.039 / 0.090 / 0.092 / 0.055 |
| LogNormal mixture | 0.124 / 0.258 / 0.400 / 0.248 | 0.043 / 0.088 / 0.074 / 0.061 |
| Joint flow | 0.101 / 0.295 / 0.618 / 0.285 | 0.046 / 0.090 / 0.124 / 0.054 |

KM T-KS is 0.116 / 0.143 / 0.408 / 0.096; Cox PH T-KS is
0.140 / 0.202 / 0.194 / 0.105. Their W1 entries are intentionally omitted.

## Table 2: informative censoring, rho=1

| Method | T-W1 | T-KS | C-KS |
|---|---:|---:|---:|
| Kaplan--Meier | -- | 0.048 +/- 0.017 | -- |
| Cox PH | -- | 0.086 +/- 0.017 | -- |
| AFT-LogNormal | 0.165 +/- 0.062 | 0.038 +/- 0.011 | -- |
| AFT-Weibull | 0.207 +/- 0.054 | 0.046 +/- 0.010 | -- |
| LogNormal mixture@IC | 0.498 +/- 0.113 | 0.103 +/- 0.017 | -- |
| Deep LogNormal--Frank | 0.339 +/- 0.113 | 0.057 +/- 0.015 | 0.067 +/- 0.015 |
| Binned KM smoother@IC | 0.527 +/- 0.138 | 0.090 +/- 0.019 | -- |
| T-only flow | 0.251 +/- 0.072 | 0.054 +/- 0.014 | -- |
| Joint flow | 0.188 +/- 0.092 | 0.039 +/- 0.013 | 0.046 +/- 0.017 |

## Table 3: semi-synthetic real covariates

| Dataset | Method | rho=0 T-KS | rho=1 T-KS |
|---|---|---:|---:|
| SUPPORT | Joint flow | 0.062 +/- 0.024 | 0.061 +/- 0.017 |
| SUPPORT | T-only flow | 0.044 +/- 0.013 | 0.066 +/- 0.028 |
| SUPPORT | Cox PH | 0.050 +/- 0.011 | 0.087 +/- 0.016 |
| Rotterdam--GBSG | Joint flow | 0.057 +/- 0.021 | 0.050 +/- 0.006 |
| Rotterdam--GBSG | T-only flow | 0.041 +/- 0.012 | 0.053 +/- 0.012 |
| Rotterdam--GBSG | Cox PH | 0.055 +/- 0.019 | 0.083 +/- 0.015 |

## Table 4: observational data

| Method | SEER C / IBS | SUPPORT C / IBS | Rotterdam--GBSG C / IBS |
|---|---|---|---|
| Joint flow | 0.882 / 0.082 | 0.578 / 0.257 | 0.640 / 0.195 |
| T-only flow | 0.887 / 0.082 | 0.591 / 0.249 | 0.656 / 0.186 |
| DeepHit | 0.884 / 0.084 | 0.605 / 0.225 | 0.628 / 0.191 |
| RSF | 0.875 / 0.083 | 0.604 / 0.221 | 0.669 / 0.179 |
| Cox PH | 0.865 / 0.092 | 0.568 / 0.235 | 0.664 / 0.184 |

Each entry in Table 4 is the mean over five 75/25 splits; the manuscript also
reports the corresponding standard deviations.
