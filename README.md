# tPowerBilby

**tPowerBilby** is a Python package based on tBilby, designed for estimating the Amplitude Spectral Density (ASD) for LIGO and VIRGO. A link to the paper that describes its methodology can be found below. 

In a nutshell, by default, the ASD data is split into several parts. Each part is treated almost independently, where a transdimensional Bayesian inference finds the best description for it. Then, during post-processing, a final posterior sample is produced. These steps are taken to maintain reasonable inference time with very high flexibility (i.e., many parameters).  

## Quick Links

- [tPowerBilby Paper](https://arxiv.org/pdf/2404.04460) - still missing
- [tBilby Paper](https://arxiv.org/pdf/2404.04460)
- [tBilby Repository](https://github.com/tBilby/tBilby.git)

## Documentation Links

- [Full list of Configuration Options](doc/README_config.md)
- [Post Processing Functionality](doc/README_postprocessing.md)

## Installation

```sh
pip install tPowerBilby  # (Note: Package not available yet)
```

## Usage

```sh
tPowerBilby config.json
```

Once done (after a few hours...), several result files will be created. To get full posterior samples, additional processing is required (in the case of multiple range inference). Please see the post-processing section for more information. 

### Example `config.json`

```json
{
    "user_label": "GW150914_post",
    "trigger_time": 1126259470.4,
    "det": "H1"
}
```
![alt text]((https://github.com/NirGutt/tPowerBilby/blob/main/tpowerbilby.png)?raw=true)



Feel free to reach out for any issues!

