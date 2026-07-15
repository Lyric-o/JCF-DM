# JCF-DM source manifest

## Main implementation

- `src/synthetic/uninformative_censoring_flow.py`: conditional monotone flow,
  Soft--Nelson--Aalen regularizer, Table 1 joint-flow runner.
- `src/synthetic/uninformative_censoring_impl.py`: uninformative DGPs,
  classical survival estimators, metrics, and plotting helpers.
- `src/synthetic/informative_censoring_impl.py`: shared-Gamma-frailty DGP,
  shared-latent joint likelihood, Table 2 joint-flow runner.
- `src/synthetic/survival_dists.py`: LogNormal, Weibull, mixture, and monotone
  spline-style survival distributions.
- `src/synthetic/competing_methods.py`: KM, Cox PH, and AFT implementations.

## Main-text comparisons

- `src/synthetic/comparison_aft_uninf.py`
- `src/synthetic/comparison_ausset.py`
- `src/synthetic/ausset_informative.py`
- `src/synthetic/comparison_classical_ic.py`
- `src/synthetic/comparison_lognormal_mixture.py`
- `src/synthetic/comparison_lognormal_mixture_ic.py`
- `src/synthetic/comparison_binned_km.py`
- `src/synthetic/comparison_binned_km_ic.py`
- `src/synthetic/comparison_deep_lognormal_frank.py`

## Clinical-covariate and observational experiments

- `src/real_data/real_data_semisynthetic.py`: Table 3.
- `src/real_data/real_data_pure.py`: Table 4.
- `src/real_data/survival_dists.py`, `src/real_data/eps_utils.py`: calibrated
  distribution and latent-sampling helpers used by those runners.

## Appendix analyses

- `src/ablations/sensitivity_lambda_ic.py`: informative-censoring
  Wasserstein-weight sweep.
- `src/ablations/sensitivity_lambda_S.py`: simulation-size sensitivity.

## Deliberately excluded

- Raw and preprocessed clinical CSV files.
- SLURM logs, caches, temporary plots, and intermediate shard files.
- Obsolete runners with the misleading method labels `DSM-IC` or `WKM-IC`.
- Manuscript TeX/PDF sources; those remain in the paper repository.
