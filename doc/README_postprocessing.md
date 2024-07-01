postprocessing options 


        
    def RunDefaultValidations(self,plot_if_negative=False):
        self.plot_maxL()
        self.plot_estimate_normal_goodness_of_fit()
        self.estimate_log_likelihood()
        self.test_estimate_log_likelihood(self.end_time+5,test_time_sec= 60,plot_if_negative=plot_if_negative)    

          Validate_class1.generate_posterior_samples(1000)
        #Validate_class1.Validate_sampled_samples('IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5')
        
