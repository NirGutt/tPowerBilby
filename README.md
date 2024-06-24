# tPowerBilby
tPowerBilby is a tBilby based python package that is intended for estimating LIGO and VIRGO Amplitude Spectral Density.  Link to the paper and to tBilby repository

Installation: 
pip install tPowerBilby (doesnt exsits yet) 

Usage:
tPowerBilby config.json



A typpical config.json looks like:
{"user_label": "GW150914_post", "trigger_time": 1126259470.4, "det": "H1"}

The full list of config.json options and their meaning (and their default value) is the following:

"use_simpler_lorenztain_in_round_0": True  // Use teh simpler version of Lorentizain in round 0, this saves time since there are less parameters to infer.      
"det":'H1'  // the interformter name 
"split_run" : True // split the runs into two rounds and multiple sections, to infer each section seperatly 
"user_label"  : 'GW150419' // output file name 
"trigger_time" : 1126259462.4 // GPS trigger time 
"outdir" : 'outdir' // output directory name 

"maximum_frequency"  : 896 // max freqency in Hz
"minimum_frequency": 20 // min freqency in Hz
"roll_off" : 0.4  # Roll off duration of tukey window in seconds, default is 0.4s

"duration": 4  # Analysis segment duration
"post_trigger_duration" = 2  # Time between trigger time and end of segment. 
Important, the end time of the asd is defined to be: trigger+post_trigger_duration-duration so the start time is trigger+post_trigger_duration-duration-duration
This way, one can provide the trigger time of the gw event

"n_exp" : 5 \\ max number of power laws  
 "n_lines" : 20 \\ max number of lines in each segment  

"min_freq_spacing":30 \\ segmentation algorithm settings. for a point to be identify as a "cut" point, it has to lay in an empty (from lines) segment of "min_freq_spacing" Hz  
"min_freq_segment" : 100 \\ the size of each segment should be larger than "min_freq_segment"

"use_low_freq_detector" : False \\ initiate a simple lines detector in the low frequency region, this is in addition to the usual lines detection algorithm  
"low_freq_limit_detector" : 40 \\ determines the frequency of which below the line detector works 









