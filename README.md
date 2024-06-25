
# tPowerBilby

**tPowerBilby** is a Python package based on tBilby, designed for estimating the Amplitude Spectral Density (ASD) for LIGO and VIRGO. 

## Quick Links

- [tPowerBilby Paper]([link/to/paper](https://arxiv.org/pdf/2404.04460))
- [tBilby Repository]([link/to/tBilby/repo](https://github.com/tBilby/tBilby.git))

## documentation Links
- [Configuration Options](doc/README_config.md)
- [Post Processing](doc/README_post_processing.md)


## Installation

```sh
pip install tPowerBilby  # (Note: Package not available yet)
```

## Usage

```sh
tPowerBilby config.json
```

### Example `config.json`

```json
{
    "user_label": "GW150914_post",
    "trigger_time": 1126259470.4,
    "det": "H1"
}
```

### Full List of `config.json` Options

- **`use_simpler_lorentzian_in_round_0`**: `True`  
  Use a simpler version of the Lorentzian in round 0 to save time with fewer parameters to infer.
  
- **`det`**: `'H1'`  
  The interferometer name.
  
- **`split_run`**: `True`  
  Split the runs into two rounds and multiple sections to infer each section separately.
  
- **`user_label`**: `'GW150419'`  
  Output file name.
  
- **`trigger_time`**: `1126259462.4`  
  GPS trigger time.
  
- **`outdir`**: `'outdir'`  
  Output directory name.

- **`maximum_frequency`**: `896`  
  Max frequency in Hz.
  
- **`minimum_frequency`**: `20`  
  Min frequency in Hz.
  
- **`roll_off`**: `0.4`  
  Roll-off duration of the Tukey window in seconds (default is 0.4s).

- **`duration`**: `4`  
  Analysis segment duration in seconds.
  
- **`post_trigger_duration`**: `2`  
  Time between the trigger time and the end of the segment.  
  Important: The end time of the ASD is defined as `trigger + post_trigger_duration - duration`, so the start time is `trigger + post_trigger_duration - duration - duration`. This way, you can provide the trigger time of the GW event.

- **`n_exp`**: `5`  
  Max number of power laws.
  
- **`n_lines`**: `20`  
  Max number of lines in each segment.

- **`min_freq_spacing`**: `30`  
  For a point to be identified as a "cut" point, it has to be in an empty (from lines) segment of `min_freq_spacing` Hz.
  
- **`min_freq_segment`**: `100`  
  Each segment should be larger than `min_freq_segment` Hz.

- **`use_low_freq_detector`**: `False`  
  Initiate a simple lines detector in the low-frequency region, in addition to the usual lines detection algorithm.
  
- **`low_freq_limit_detector`**: `40`  
  Determines the frequency below which the line detector works.

## Notes

- Make sure to adjust the configuration options according to your analysis needs.
- The package is still under development, so some bugs may appear. 

Feel free to reach out for any issues!








