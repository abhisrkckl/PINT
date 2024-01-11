# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved `get_derived_params` to `timing_model`
- `check_ephemeris_connection` CI test no longer requires access to static NANOGrav site
- `TimingModel.compare()` now calls `change_binary_epoch()`.
### Added
- Added numdifftools to setup.cfg to match requirements.txt
- Documentation: Added `convert_parfile` to list of command-line tools in RTD
- DDH binary model
- function `pint.utils.xxxselections` to do DMX-style selections for any parameter name
- Plot model DM in pintk
- More tests for pintk
- Maximum likelihood fitting for ECORR
    - `is_time_correlated` class attribute in correlated `NoiseComponent`s
    - `has_time_correlated_errors` property in `TimingModel`
    - `Residuals._calc_ecorr_chi2()` method for fast chi2 computation using Sherman-Morrison identity
    - `pint.utils.sherman_morrison_dot` and `pint.utils.woodbury_dot`
    - Refactored repeated code out of `Residuals.calc_phase_mean` and `Residuals.calc_time_mean`
    - Simplified `Residuals._calc_gls_chi2()` so that it uses Woodbury identity directly
    - Refactored WLS chi2 code out of `Residuals.calc_chi2()` into a new function `Residuals._calc_wls_chi2()`
    - `Residuals.d_lnlikelihood_d_whitenoise_param` will throw a `NotImplementedError` when correlated noise is present.
    - `DownhillFitter._fit_noise()` doesn't use derivatives when correlated noise is present.
    - Documentation: Noise fitting example notebook.
### Fixed
- `MCMC_walkthrough` notebook now runs
- Fixed runtime data README 
- Fixed `derived_params` when OMDOT has 0 uncertainty
- `model.find_empty_masks` will now also look at DMX and SWX parameters
- Fixed `make_fake_toas_fromtim`
- Better exceptions for unsupported/unimplemented binary models (BTX, MSS, etc.)
- Use `Hessian` instead of `Hessdiag` in `DownhillFitter._fit_noise`; compute noise parameter uncertainties only once in `DownhillFitter.fit_toas`.
- Consistent naming in `TimingModel.get_params_mapping()`
### Removed
