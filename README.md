# tPowerBilby

**tPowerBilby** is a Python package based on tBilby, designed for estimating the Amplitude Spectral Density (ASD) for LIGO and VIRGO. A link to the paper that describes its methodology can be found below. 

In a nutshell, by default, the ASD data is split into several parts. Each part is treated almost independently, where a transdimensional Bayesian inference finds the best description for it. Then, during post-processing, a final posterior sample is produced. These steps are taken to maintain reasonable inference time with very high flexibility (i.e., many parameters).  

Please note that the code is still undergoing an improvement phase, with further refinements and enhancements expected in the future.

## Quick Links

- [tPowerBilby Paper](https://arxiv.org/abs/2501.03285)
- [tBilby Paper](https://arxiv.org/pdf/2404.04460)
- [tBilby Repository](https://github.com/tBilby/tBilby.git)

## Documentation Links

- [Full list of Configuration Options](doc/README_config.md)


## Installation

1. Install **tbilby**. 
2. Download the package and work within the local directory, as this code does not provide functionality for external use.

## Usage

```sh
python tPowerBilby.py config.json
```

Once done (after a few hours...), several result files will be created. Some are simple images used for validation. Other contain the ASD estimations. Please see the post-processing section for more information. 

### Example `config.json`

```json
{
    "user_label": "GW150914_post",
    "trigger_time": 1126259470.4,
    "det": "H1"
}
```
### Code Architecture

Here is how the code is structured in case you want to take a look under the hood.

![alt text](https://github.com/NirGutt/tPowerBilby/blob/main/tpowerbilby.png)



Feel free to reach out for any issues!

