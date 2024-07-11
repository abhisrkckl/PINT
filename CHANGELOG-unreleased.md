# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- Type hints in `pint.derived_quantities`
- `plrednoise_from_wavex()` and `pldmnoise_from_dmwavex()` functions now compute `TNRedFLow` and `TNDMFLow`
- `powerlaw_corner` function
- `TNREDFLOW` and `TNREDCORNER` parameters in `PLRedNoise`
- `TNDMFLOW` and `TNDMCORNER` parameters in `PLDMNoise`
### Fixed
### Removed
- Unnecessary default arguments from the `powerlaw()` function.