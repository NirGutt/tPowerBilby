import tpowerbilby.postprocessing as postprocessing
import matplotlib.pyplot as plt
plt.close('all')


Validate_class = postprocessing.ValidateAndGenerateASD(det='H1',folder='H1_post/',
                                                  sampling_label = 'GW150914_post',trigger_time = 1126259470.4)

Validate_class.RunDefaultValidations()    
Validate_class.generate_posterior_samples(1000)
Validate_class.Validate_sampled_samples('IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5')
    
    
