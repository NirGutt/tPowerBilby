# Standardized Configuration Parameters

This document provides an explanation of the configuration parameters used in the analysis. 

## General Configuration
- **`det`**: Specifies the detector to use (e.g., `'H1'` for LIGO Hanford).
- **`trigger_time`**: The GPS time of the event trigger.
- **`maximum_frequency`**: Maximum frequency in Hz for the analysis range.
- **`minimum_frequency`**: Minimum frequency in Hz for the analysis range.
- **`roll_off`**: Roll-off duration of the Tukey window (in seconds), default is `0.4s`.
- **`duration`**: Total duration (in seconds) of the analysis segment.
- **`frequency_resolution`**: Frequency resolution, default is `1 / duration`, but can be modified to match a desired resolution since a continuous model is used (not fully tested).
- **`post_trigger_duration`**: Time (in seconds) between the trigger time and the end of the segment.

## Pre-Processing Settings
- **`pre_processing_n_looking_back`**: Number of segments to look back during preprocessing.
- **`min_freq_segment`**: Minimum frequency range (in Hz) for each segment.
- **`low_freq_limit_detector`**: Below this value, RANSAC analysis is used to detect the lines.
- **`pre_processing_end_time`**: The end time for preprocessing.

## Sampling Settings
- **`n_exp`**: Number of exponentials used for sampling.
- **`n_lines`**: Number of spectral lines to include in the model for each segment.
- **`user_label`**: User-defined label for the analysis (e.g., `'GW150419'`).
- **`outdir`**: Directory where output files will be stored.
- **`resume`**: Boolean indicating whether to resume a previous run.
- **`fit_entire_data`**: If `false`, low-quality frequency bins are removed, creating a hybrid between Welch and tBilby methods.
- **`N_noise_samples`**: Number of noise samples to generate during the analysis.
- **`skip_samples_writing`**: Boolean indicating whether to skip writing samples to disk.
- **`N_live_points`**: Number of live points used in the sampling process.

## Debugging
- **`debug`**: Boolean enabling debugging mode for additional logging and diagnostics.

## Notes
- Ensure all required parameters are correctly set before running the analysis.
- Modify paths and user-specific settings (e.g., `outdir` and `user_label`) as needed.
