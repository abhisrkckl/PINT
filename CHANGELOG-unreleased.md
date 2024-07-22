# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved the events -> TOAs and photon weights code into the function `load_events_weights` within `event_optimize`.
- Updated the `maxMJD` argument in `event_optimize` to default to the current mjd
### Added
- Type hints in `pint.derived_quantities`
- Doing `model.par = something` will try to assign to `par.quantity` or `par.value` but will give warning
- `plrednoise_from_wavex()` and `pldmnoise_from_dmwavex()` functions now compute `TNRedFLow` and `TNDMFLow`
- `powerlaw_corner` function
- `TNREDFLOW` and `TNREDCORNER` parameters in `PLRedNoise`
- `TNDMFLOW` and `TNDMCORNER` parameters in `PLDMNoise`
- `PLChromNoise` component to model chromatic red noise with a power law spectrum
### Fixed
- Explicit type conversion in `woodbury_dot()` function
- Documentation: Fixed empty descriptions in the timing model components table
### Removed
- Removed the argument `--usepickle` in `event_optimize` as the `load_events_weights` function checks the events file type to see if the 
file is a pickle file.
- Removed obsolete code, such as manually tracking the progress of the MCMC run within `event_optimize`
- Unnecessary default arguments from the `powerlaw()` function.
