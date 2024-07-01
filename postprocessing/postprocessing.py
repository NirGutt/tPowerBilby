import bilby 
from context import tbilby
from gwpy.timeseries import TimeSeries
from scipy import stats
from bilby.core.utils import nfft 
from scipy import fft as sp_fft
from bilby.core.prior import Prior, Uniform, Interped
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import ConditionalLogUniform, LogUniform,Uniform
from scipy import interpolate
import arviz as az
import inspect
from scipy.signal import find_peaks_cwt
from scipy import signal
from gwpy.timeseries import TimeSeries
import matplotlib.colors as mcolors
import pickle
from bilby.core.utils import infer_parameters_from_function
import gwpy
import asd_data_manipulations 

#########CONFIGURATION


detectors = ["H1", "L1","V1"]

class ValidateAndGenerateASD:
    def __init__(self, det,folder,  sampling_label,trigger_time,
               minimum_frequency=20,maximum_frequency=896,roll_off=0.4,duration=4,post_trigger_duration=2,HQ_n_sigma=4):
        self.result_fnames=[]  
        self.det = det
        self.sampling_label = sampling_label
        self.HQ_n_sigma=HQ_n_sigma
        self.trigger_time = trigger_time        
        self.pre_folder=folder+'/'+self._extract_single_prefix(folder)

        self.range_vec = self._extract_numbers_from_filenames(folder)
        print(self.range_vec)  
        for i in np.arange(len(self.range_vec)-1):
             self.result_fnames.append(self.pre_folder+str(det)+'_round_1_'+str(self.range_vec[i])+'_'+str(self.range_vec[i+1])+'_result.json')
    
        self.fname_full_range= self.pre_folder+str(det)+'_round_0_'+str(self.range_vec[0])+'_'+str(self.range_vec[-1])+'_full_range_result.json'
        self.minimum_frequency=minimum_frequency
        self.maximum_frequency=maximum_frequency
        self.roll_off = roll_off
        self.duration = duration
        self.post_trigger_duration=post_trigger_duration
        self.end_time = trigger_time + post_trigger_duration
        self.start_time = self.end_time - duration
        self.best_function_fit=None
        self.I_keep_bins=None
    
        self._create_models()

        self.x,self.y = self._get_GW_data()    
        
    def _extract_numbers_from_filenames(self,folder_path):
        import re
        import os
    # Regular expression to match the numbers in the filenames
        pattern = re.compile(r'_(\d+\.\d+)_([\d.]+)_')
        
        # Set to store unique numbers
        unique_float_numbers = set()
        
        # List all files in the folder
        for filename in os.listdir(folder_path):
            # Search for the pattern in the filename
            match = pattern.search(filename)
            if match:
                # Extract the numbers and add them to the set
                number1 = match.group(1)
                if '.' in number1:
                    unique_float_numbers.add(float(number1))
                
                # Extract the second number and check if it's a float
                number2 = match.group(2)
                if '.' in number2:
                    unique_float_numbers.add(float(number2))
        
        # Convert the set to a sorted list
        sorted_unique_numbers = sorted(unique_float_numbers)
        
        return sorted_unique_numbers    
    
    def _extract_single_prefix(self,folder_path):
        import re
        import os
    # Regular expression to match the prefix up to H1, L1, or V1
        pattern = re.compile(r'^(.*?)(_H1|_L1|_V1)_')
        #pattern = re.compile(r'^(.*?)_(H1|L1|V1)_')
    
        # Variable to store the prefix
        prefix = None
        # List all files in the folder
        for filename in os.listdir(folder_path):
            # Search for the pattern in the filename
            match = pattern.search(filename)
            if match:
                current_prefix = match.group(1)
                if prefix is None:
                    prefix = current_prefix
                elif prefix != current_prefix:
                    raise ValueError(f"File {filename} has a different prefix ({current_prefix}) than expected ({prefix})")
    
        if prefix is None:
            raise ValueError("No valid files found in the directory")
    
        return prefix+'_'

    def _create_models(self):
        self.n_exp=5
        self.n_peaks=20#est_peaks+5
        self.n_shaplets0=5
        self.n_shaplets1=5
        self.n_shaplets2=5
        self.n_shaplets3=5
        
        
        self.componant_functions_dict={}
        self.componant_functions_dict[self.exp]=(self.n_exp,'A','lamda')
        self.componant_functions_dict[self.improved_lorentzian]=(self.n_peaks,'loc', 'Amp', 'gam','zeta','tau')
        self.componant_functions_dict[self.shaplets0]=(self.n_shaplets0,'s0Amp', True,'n_shaplets0')
        self.componant_functions_dict[self.shaplets1]=(self.n_shaplets1,'s1Amp', True,'n_shaplets1')
        self.componant_functions_dict[self.shaplets2]=(self.n_shaplets2,'s2Amp', True,'n_shaplets2')
        self.componant_functions_dict[self.shaplets3]=(self.n_shaplets3,'s3Amp', True,'n_shaplets3')
        
        
        self.model = tbilby.core.base.create_transdimensional_model('model',  self.componant_functions_dict,returns_polarization=False,SaveTofile=False)
        
        
        self.componant_functions_dict_s={}
        self.componant_functions_dict_s[self.exp]=(self.n_exp,'A','lamda')
        self.componant_functions_dict_s[self.lorentzian]=(self.n_peaks,'loc', 'Amp', 'gam')
        self.componant_functions_dict_s[self.shaplets0]=(self.n_shaplets0,'s0Amp', True,'n_shaplets0')
        self.componant_functions_dict_s[self.shaplets1]=(self.n_shaplets1,'s1Amp', True,'n_shaplets1')
        self.componant_functions_dict_s[self.shaplets2]=(self.n_shaplets2,'s2Amp', True,'n_shaplets2')
        self.componant_functions_dict[self.shaplets3]=(self.n_shaplets3,'s3Amp', True,'n_shaplets3')
        
        self.model_Lorentz_simpler = tbilby.core.base.create_transdimensional_model('model_Lorentz_simpler',  self.componant_functions_dict_s,returns_polarization=False,SaveTofile=False)
    
    def _identify_model(self,params):
        model_needed_params = infer_parameters_from_function(self.model)
        simpler_model_needed_params = infer_parameters_from_function(self.model_Lorentz_simpler)
        if all(p in params for p in model_needed_params):
            return self.model,self.componant_functions_dict
        
        if all(p in params for p in simpler_model_needed_params):
            return self.model_Lorentz_simpler,self.componant_functions_dict
        
        
        return np.nan

    def exp(self,x,A,lamda):  
      if isinstance(x,np.ndarray):  
          xfunc = x.copy() # start from zero offset 
      else:
          xfunc= x
      return A*np.power(xfunc,lamda)
    
    
    def improved_lorentzian(self,x, loc, Amp, gam,zeta,tau):
   
      
        damped_alpha = np.zeros(x.shape) 
        df_i = zeta *gam 
        I_left_tail = x < loc - df_i
        I_right_tail =x > loc + df_i
        
        damped_alpha [I_right_tail | I_left_tail] = tau 
    
        return np.exp(-damped_alpha*(np.abs(x-(loc))-df_i)) *Amp * gam**2 / ( gam**2 + ( x - loc )**2)
    
    
    
    def lorentzian(self,x, loc, Amp, gam):
        # damped tails lorentzian, after 3sigma start dying out 
        decay =1
        damped_alpha = np.zeros(x.shape) 
        df_i = loc/50
        I_left_tail = x < loc - df_i
        I_right_tail =x > loc + df_i
       
        damped_alpha [I_right_tail | I_left_tail] = decay 
        return np.exp(-damped_alpha*(x-df_i)/(df_i)) *Amp * gam**2 / ( gam**2 + ( x - loc )**2)
    
    
    def gauss(self,x,Ag,mu,sigma):  
      xfunc = x.copy() # start from zero offset 
      return Ag*np.exp(-((xfunc-mu)/sigma)**2)
    
    
    def shaplet_func(self,x_in, deg =0 ,beta=1):
        x=x_in/beta
        coef= np.zeros((deg+1,))
        coef[deg]=1
          
        n = deg
        
        const =   1/np.sqrt(2**n * np.pi**(0.5) * np.math.factorial(n)) 
        weight = np.exp(-0.5*x**2) 
        poly = np.polynomial.hermite.Hermite(coef=coef.astype(int))(x)
        
        vals = const*poly*weight/np.sqrt(beta)
          
        return vals 
    
    def shaplets0(self,x,n_shaplets0,s0x_center,s0beta,s0Amp):
       xfunc = x.copy()
        
       return s0Amp*self.shaplet_func(xfunc-s0x_center,n_shaplets0,s0beta)
    
    def shaplets1(self,x,n_shaplets1,s1x_center,s1beta,s1Amp):
       xfunc = x.copy()
        
       return s1Amp*self.shaplet_func(xfunc-s1x_center,n_shaplets1,s1beta)
    
    
    def shaplets2(self,x,n_shaplets2,s2x_center,s2beta,s2Amp):
       xfunc = x.copy()
        
       return s2Amp*self.shaplet_func(xfunc-s2x_center,n_shaplets2,s2beta)
    
    
    def shaplets3(self,x,n_shaplets3,s3x_center,s3beta,s3Amp):
       xfunc = x.copy()
        
       return s3Amp*self.shaplet_func(xfunc-s3x_center,n_shaplets3,s3beta)

    def _log_l_arr(self,data,asd_est):
        print('log_l_arr')
     
        arr= -0.5* (4/self.duration)*(data/asd_est)**2 -2*np.log(asd_est)
        print(np.sum(arr))
        return arr  
 
    def _stand_alone_log_l(self,data,asd_est):
        
        log_l =  np.sum(-0.5* (4/self.duration)*(data/asd_est)**2 -2*np.log(asd_est))

        return round(log_l, 2)

    def _log_l_dist(self,x,median_arr,fit_arr,I_keep_bins,start_time,test_secs=10,plot_if_negative=False ):
        
        log_l_list=[]
        for t in np.arange(start_time,start_time+test_secs,self.duration):
            
            print(t)
            x_check,y_check = self._get_GW_data_by_trigger(start_time=t,psd_duration=self.duration)
        
        
            log_l_median_HQ = self._stand_alone_log_l(y_check[I_keep_bins],median_arr[I_keep_bins])
            log_l_fit_HQ = self._stand_alone_log_l(y_check[I_keep_bins],fit_arr[I_keep_bins])
            
            log_l_median_LQ = self._stand_alone_log_l(y_check,median_arr)
            log_l_fit_LQ = self._stand_alone_log_l(y_check,fit_arr)
            
            
            if plot_if_negative and log_l_fit_HQ-log_l_median_HQ<0:
                plt.figure()
                plt.subplot(2,1,1)
                plt.loglog(x_check,y_check)
                plt.loglog(x,fit_arr,'-k')
                plt.loglog(x,median_arr,'m')
                plt.subplot(2,1,2)
                plt.semilogx(x[I_keep_bins],np.cumsum(self._log_l_arr(y_check[I_keep_bins],fit_arr[I_keep_bins]))-np.cumsum(self._log_l_arr(y_check[I_keep_bins],median_arr[I_keep_bins])),'-k',label = 'HQ')
                plt.semilogx(x,np.cumsum(self._log_l_arr(y_check,fit_arr))-np.cumsum(self._log_l_arr(y_check,median_arr)),'-r',label = 'LQ')
     
            
            
            log_l_list.append([log_l_median_HQ,log_l_fit_HQ,log_l_median_LQ,log_l_fit_LQ ])
            
            print(log_l_fit_HQ-log_l_median_HQ)
            print('--------')
            
        df = pd.DataFrame(data=log_l_list,columns=['log_l_median_HQ','log_l_fit_HQ','log_l_median_LQ','log_l_fit_LQ' ])    
        plt.figure()
       
        plt.hist(df['log_l_fit_HQ']-df['log_l_median_HQ'],label='$\Delta \log \mathcal{L}_{HQ}$',bins=20,color='indigo')   
        plt.hist(df['log_l_fit_LQ']-df['log_l_median_LQ'],label='$\Delta \log \mathcal{L}_{LQ}$',bins=20,alpha=0.8,color='r')
        plt.legend()
      
        
    def _log_l_dist_mixture(self,x,median_arr,fit_arr0,fit_arr1,I_keep_bins0,I_keep_bins1,start_time,lambda0=0.5,test_secs=10 ):

          I_keep_bins =I_keep_bins0& I_keep_bins1          
          
          
          log_l_list=[]
          for t in np.arange(start_time,start_time+test_secs,self.duration):
              
              print(t)
              x_check,y_check = self._get_GW_data_by_trigger(start_time=t,psd_duration=self.duration)
          
          
              log_l_median_HQ = self._stand_alone_log_l(y_check[I_keep_bins],median_arr[I_keep_bins])
              log_l_fit_HQ_0 = np.log(lambda0) + self._stand_alone_log_l(y_check[I_keep_bins],fit_arr0[I_keep_bins])
              log_l_fit_HQ_1 = np.log(1-lambda0) + self._stand_alone_log_l(y_check[I_keep_bins],fit_arr1[I_keep_bins])
              
              log_l_fit_HQ= np.logaddexp(log_l_fit_HQ_0,log_l_fit_HQ_1)
              
              log_l_median_LQ = self._stand_alone_log_l(y_check,median_arr)
              log_l_fit_LQ_0 = np.log(lambda0) + self._stand_alone_log_l(y_check,fit_arr0)
              log_l_fit_LQ_1 = np.log(1-lambda0) + self._stand_alone_log_l(y_check,fit_arr1)
              
              log_l_fit_LQ= np.logaddexp(log_l_fit_LQ_0,log_l_fit_LQ_1)
              
          
              log_l_list.append([log_l_median_HQ,log_l_fit_HQ,log_l_median_LQ,log_l_fit_LQ ])
              
              print(log_l_fit_HQ-log_l_median_HQ)
              print('--------')
              
          df = pd.DataFrame(data=log_l_list,columns=['log_l_median_HQ','log_l_fit_HQ','log_l_median_LQ','log_l_fit_LQ' ])    
          plt.figure()
         
          plt.hist(df['log_l_fit_HQ']-df['log_l_median_HQ'],label='$\Delta \log \mathcal{L}_{HQ}$',bins=20,color='indigo')   
          plt.hist(df['log_l_fit_LQ']-df['log_l_median_LQ'],label='$\Delta \log \mathcal{L}_{LQ}$',bins=20,alpha=0.8,color='r')
          plt.legend()


###################

    def _extract_posteriors_from_files(self,file_list):
          
        dict_results_posteriors={}    
        for file in file_list:        
            result = tbilby.core.base.result.read_in_result(filename=file)        
            if 'full_range' in file:    
                dict_results_posteriors['BroadBand']=result.posterior
            else:
                dict_results_posteriors[file]=result.posterior
            
        return dict_results_posteriors
    
    
    def _merge_and_generate_samples(self,file_list,fname_full_range,number_of_samples,x,genrate_mask):
        
        file_list.append(fname_full_range)
        
        collected_data = {}

        def save_asds_to_pickle(filename):
            values = list(collected_data.values())  # Extract values from the dictionary
            with open(filename, 'wb') as f:
                pickle.dump(values, f)
        
    
        def add_or_sum(location, value):
            if location in collected_data:
                collected_data[location] += value
            else:
                collected_data[location] = value
        
        dict_results_posteriors = self._extract_posteriors_from_files(file_list)        
        
        
        
        for fit_part in dict_results_posteriors.keys():        
            
            model,componant_functions_dict = self._identify_model(dict_results_posteriors[fit_part].iloc[0])
            needed_params = infer_parameters_from_function(model)
            
            borad_band=False        
            if str(fit_part)=='BroadBand':
                borad_band=True       
                 
            drop_this = []
            part_samples = dict_results_posteriors[fit_part].sample(number_of_samples)
            for col in part_samples.columns:
                if col not in needed_params:
                    drop_this.append(col)
            part_samples.drop(columns = drop_this,inplace=True)  
            
            if number_of_samples < len(part_samples):
                print(fit_part +' doesnt contian enough entries, requested ' + str(number_of_samples) + 'but contains ' + str(len(part_samples)) )
                fffff
                
            # randomize it     
            part_samples = part_samples.sample(frac = 1)
            
            for sample_i in np.arange(number_of_samples):             
                model_parameters =part_samples.iloc[sample_i].to_dict()   
                
              
                if not borad_band:# set the power laws to zero 
                    model_parameters['n_exp']=0
                
                est_asd = model(x,**model_parameters)
                add_or_sum(str(sample_i),est_asd)
                
        # save the data into file 
        
        
        
        save_asds_to_pickle(self.sampling_label+'asd_samples_'+self.det+'.pkl')
        
        if genrate_mask:
            I_hq = self.get_HQ_index()
   
            with open(self.sampling_label+'_mask_asd_samples_'+self.det+'.pkl', 'wb') as f:
                pickle.dump(I_hq, f)
            plt.figure()
            plt.loglog(self.x,self.y,'-c')    
            plt.loglog(self.x[~I_hq],self.y[~I_hq],'or') 
        
        
        
        return collected_data



        
    
   
    
    def _get_GW_data_meidan(self):
        
        
    
        psd_duration = 32 * self.duration
        psd_start_time = self.start_time - psd_duration
        psd_end_time = self.start_time
        

        psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(self.det, psd_start_time, psd_end_time)
        
    
        psd_alpha = 2 * self.roll_off / self.duration
        psd = psd_data.psd(
            fftlength=self.duration, overlap=0, window=("tukey", psd_alpha), method="median"
        )
       
        
        psd_frequencies = psd.frequencies.value
        psd = psd.value
        I = (psd_frequencies >= self.minimum_frequency) &  (psd_frequencies <= self.maximum_frequency) 
        
        
        return psd_frequencies[I],np.sqrt(psd[I])
     
    
    
    def _get_GW_data_by_trigger(self,start_time,psd_duration, f_i=None,f_f=None,return_raw_data =False):
        

    
    
        if f_i is None:
            f_i =self.minimum_frequency
        if f_f is None:
            f_f=self.maximum_frequency  
    
        # making sure we are at a reasnable range 
        if self.minimum_frequency > f_i:
            f_i ==self.minimum_frequency 
        if self.maximum_frequency < f_f:
            f_f = self.maximum_frequency 
    
       
        psd_start_time = start_time - psd_duration
        psd_end_time = start_time
        
        
     
        psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(self.det, psd_start_time, psd_end_time)
        ifo = bilby.gw.detector.get_empty_interferometer(self.det)    
        ifo.strain_data.set_from_gwpy_timeseries(psd_data)
        ifo.maximum_frequency = self.maximum_frequency
        ifo.minimum_frequency = self.minimum_frequency
        
        
        x_ifo = ifo.strain_data.frequency_array
        y_ifo = ifo.strain_data.frequency_domain_strain
        
        I = (x_ifo >= f_i) &  (x_ifo <= f_f) 
    
        if return_raw_data:
            return x_ifo[I],y_ifo[I]
        
        return x_ifo[I],np.abs(y_ifo[I])
    
    
    def _get_GW_local_psd_data(self,f_i=None,f_f=None):
        


        if f_i is None:
            f_i =self.minimum_frequency
        if f_f is None:
            f_f=self.maximum_frequency  

        # making sure we are at a reasnable range 
        if self.minimum_frequency > f_i:
            f_i =self.minimum_frequency 
        if self.maximum_frequency < f_f:
            f_f = self.maximum_frequency 

        psd_duration = 4 
        psd_start_time = self.start_time - psd_duration
        psd_end_time = self.start_time
        
        
  
        psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(self.det, psd_start_time, psd_end_time)
        
      
        psd_alpha = 2 * self.roll_off / self.duration
        psd = psd_data.psd(
            fftlength=psd_duration, overlap=0, window=("tukey", psd_alpha), method="median"
        )
      
        psd_frequencies = psd.frequencies.value
        psd = psd.value
        I = (psd_frequencies >= f_i) &  (psd_frequencies <= f_f) 
        
        
        return psd_frequencies[I],np.sqrt(psd[I])
    
    def _get_GW_data(self,f_i=None,f_f=None,return_raw_data =False):
    

        
        if f_i is None:
            f_i =self.minimum_frequency
        if f_f is None:
            f_f=self.maximum_frequency  
    
        # making sure we are at a reasnable range 
        if self.minimum_frequency > f_i:
            f_i =self.minimum_frequency 
        if self.maximum_frequency < f_f:
            f_f = self.maximum_frequency 
    
        
        psd_duration = self.duration
        psd_start_time = self.start_time - psd_duration
        psd_end_time = self.start_time
        
        psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(self.det, psd_start_time, psd_end_time)       
        ifo = bilby.gw.detector.get_empty_interferometer(self.det)    
        ifo.strain_data.set_from_gwpy_timeseries(psd_data)
        ifo.maximum_frequency = self.maximum_frequency
        ifo.minimum_frequency = self.minimum_frequency
        
        
        x_ifo = ifo.strain_data.frequency_array
        y_ifo = ifo.strain_data.frequency_domain_strain
        
        I = (x_ifo >= f_i) &  (x_ifo <= f_f) 
    
        if return_raw_data:
            return x_ifo[I],y_ifo[I]
        
        return x_ifo[I],np.abs(y_ifo[I])
        
    
    
     
    def get_HQ_index(self):
        if self.I_keep_bins is None:
            self.plot_maxL(plotit=False)
                        
        return self.I_keep_bins 


    def get_max_likelihood_curve(self):
        if self.best_function_fit is None:
            self.plot_maxL(plotit=False)
                        
        return  self.best_function_fit   
        
        

    def plot_maxL(self,plotit=True):

        if plotit:
            plt.figure()    
            plt.loglog(self.x,self.y,label='data')
        result_fname='';

        colors =mcolors.CSS4_COLORS
        
        x_best=[]
        asd_est_best=[]
        smooth_curve_best=[]
        exp_smooth_curve_best=[]
        n_lines=[]
        
        
        
        discrete_vars_vals= []
        
        result = tbilby.core.base.result.read_in_result(filename=self.fname_full_range)
        model,componant_functions_dict = self._identify_model(result.posterior.iloc[0].to_dict())
        result,cols = tbilby.core.base.preprocess_results(result,componant_functions_dict,remove_ghost_samples=False,return_samples_of_most_freq_component_function=True)
        best_params_post = tbilby.core.base.extract_maximal_likelihood_param_values(result=result)
        
        needed_params = infer_parameters_from_function(model)
        model_parameters = {k: best_params_post[k] for k in needed_params}
        best_function_fit = model(self.x,**model_parameters)
        best_function_fit_smooth = model(self.x,**model_parameters)
        discrete_vars_vals.append([model_parameters['n_exp']])
        
       


        for j,c in zip(np.arange(len(self.range_vec)-1),list(colors.keys())[20:20+len(self.range_vec)-1]):
            f_i = self.range_vec[j]
            f_f = self.range_vec[j+1]
            result_fname=self.result_fnames[j]
            
            x,y = self._get_GW_data(f_i,f_f)
        
        
            print('working on ' + result_fname )
            result = tbilby.core.base.result.read_in_result(filename=result_fname)
            model,componant_functions_dict = self._identify_model(result.posterior.iloc[0].to_dict())

            
            result,cols = tbilby.core.base.preprocess_results(result,componant_functions_dict,remove_ghost_samples=False,return_samples_of_most_freq_component_function=True)
            best_params_post = tbilby.core.base.extract_maximal_likelihood_param_values(result=result)
           
            needed_params = infer_parameters_from_function(model)
            model_parameters = {k: best_params_post[k] for k in needed_params}
            est_asd = model(x,**model_parameters)
            
            discrete_vars_vals.append([model_parameters['n_exp'],model_parameters['n_improved_lorentzian'],\
                                       model_parameters['n_shaplets0'],\
                                       model_parameters['n_shaplets1'],\
                                       model_parameters['n_shaplets2'],\
                                       model_parameters['n_shaplets3']]
                                      )
            
            model_parameters_no_smooth = model_parameters.copy()
            for k in np.arange(self.n_exp):
                model_parameters_no_smooth['A'+str(k)]=0 # set the amplitude to zero 
            
            best_function_fit+=model(self.x,**model_parameters_no_smooth)
            # set the number of peaks 
            n_lines.append(model_parameters['n_improved_lorentzian'])
            model_parameters['n_improved_lorentzian']=0
        
            model_parameters['n_shaplets0']=0
            model_parameters['n_shaplets1']=0
            model_parameters['n_shaplets2']=0
            model_parameters['n_shaplets3']=0
            exp_smooth_curve_best.append(model(x,**model_parameters))
            
            
            x_best.append(x)
            asd_est_best.append(est_asd)
            
            
        
            
            for n in np.arange(0):
                params = result.posterior.sample(1).iloc[0].to_dict()        
                model_parameters_t = {k: params[k] for k in needed_params}
                est_asd = model(x,**model_parameters_t)
                if plotit:
                    plt.loglog(x,est_asd,'-k',alpha=0.5)
        
        if plotit:
            
            
            x_psd_local, freq_psd_local = self._get_GW_local_psd_data()
            x_med, freq_med = self._get_GW_data_meidan()    
            plt.loglog(x_med, freq_med,'-',color = 'r',alpha=0.7,label='median')
            plt.loglog(x_psd_local, freq_psd_local,'-',color = 'm',alpha=0.5,label='local psd')
            plt.loglog(x_psd_local,best_function_fit,'-k',label='max likelihood fit')
            plt.legend()
        
        
        
       
        data_driven_HQ = (self.y * 2/np.sqrt(self.duration)) <= (best_function_fit_smooth*self.HQ_n_sigma)
        
        self.I_keep_bins = (best_function_fit <= (best_function_fit_smooth*self.HQ_n_sigma)) | data_driven_HQ
        self.I_remove_bins = best_function_fit > (best_function_fit_smooth*self.HQ_n_sigma)
        
        
        if plotit:
            plt.figure()
            plt.loglog(self.x,self.y,'c',label='HQ data')
            plt.loglog(self.x[~self.I_keep_bins],self.y[~self.I_keep_bins],'mo',label='HQ data')
            plt.loglog(self.x,best_function_fit,'k')
            plt.loglog(self.x,best_function_fit_smooth,'-.r')
         
        
        
        
        print('discrete params vals')
        print(discrete_vars_vals)
        
        self.best_function_fit=best_function_fit
        
        
        
    def plot_estimate_normal_goodness_of_fit(self):
            best_curve= self.get_max_likelihood_curve()
            
            
            x_med, freq_med = self._get_GW_data_meidan()
            print('getting data nad detrending (removing mean )')
            freq_arr, psd_data = self._get_GW_data(return_raw_data =True)
            print('mean value, should be zero' + str(np.mean(psd_data)))
       
            is_it_normal = np.real(psd_data)/best_curve
            is_it_normal_im = np.imag(psd_data)/best_curve
            
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(self.x,is_it_normal,label='whiten data real')
            plt.legend()
            pval_real = stats.kstest(is_it_normal,stats.norm.cdf).pvalue
            pval_im = stats.kstest(is_it_normal_im,stats.norm.cdf).pvalue
            pval_real_med = stats.kstest(np.real(psd_data)/freq_med,stats.norm.cdf).pvalue
            plt.title('p val real = ' + str(round(pval_real,2)) +  ' p val im = ' + str(round(pval_im,2)) + ' p val med real' + str(round(pval_real_med,2)))
            
            
            
        
           
            plt.subplot(2,1,2)
            mu = 0
            variance = 1
            sigma = np.sqrt(variance)
            bins = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)
            n,bins,patches = plt.hist(is_it_normal,bins,log=True,color='k',alpha=0.7,label='fit Whiten real value',histtype='step')
            n,bins,patches = plt.hist(is_it_normal_im,bins=bins,log=True,color='m',alpha=0.7,label='fit Whiten imag value')
            
            
            n,bins,patches = plt.hist(np.real(psd_data)/freq_med,bins=bins,log=True,alpha=0.4,color='c',label='median Whiten data')
           
            
            x_guass = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)
            plt.semilogy(x_guass, (bins[1]-bins[0])*len(psd_data)*stats.norm.pdf(x_guass, mu, sigma),label='N(0,1)')
            legend =plt.legend()
            
            
            legend.get_frame().set_alpha(0)
            legend.get_frame().set_facecolor((0, 0, 1, 0.1))
            
            
            
            plt.figure(figsize=(10,10))
            mu = 0
            variance = 1
            sigma = np.sqrt(variance)
            bins = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
            
            
           
            n,bins,patches = plt.hist(np.real(psd_data)/freq_med,bins=bins,log=True,alpha=0.4,color='k',label='median Whiten real value',edgecolor ='none',histtype='stepfilled')
            
            n,bins,patches = plt.hist(is_it_normal,bins,log=True,color='k',alpha=0.99,label='ASD Whiten real value',histtype='step')
            n,bins,patches = plt.hist(is_it_normal_im,bins=bins,log=True,color='r',alpha=0.99,label='ASD Whiten imaginary value',histtype='step')
            
          
            
            x_guass = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
            plt.semilogy(x_guass, (bins[1]-bins[0])*len(psd_data)*stats.norm.pdf(x_guass, mu, sigma),label='N(0,1)',color='blue')
            legend =plt.legend()
            
            
            legend.get_frame().set_alpha(0)
            legend.get_frame().set_facecolor((0, 0, 1, 0.1))
            
            plt.grid(False)
            
          
            
            plt.ylabel('Counts',fontsize=14)
            plt.xlabel('Whitened Data $d/ASD$',fontsize=14)
             
            plt.legend(framealpha=0.0,frameon=False, ncol=1,loc="lower center",fontsize=14)
            #plt.grid(True,linestyle='--')
            plt.savefig(self.sampling_label+'_'+ self.det +'_'  +'tbilby_normal_dist_sampling.pdf',format='pdf',dpi= 300,bbox_inches="tight")
           
            

        
    def estimate_log_likelihood(self,log=True,x_compare_to_psd=None,psd_compare_to_psd=None):
        best_curve= self.get_max_likelihood_curve()
        x_med, freq_med = self._get_GW_data_meidan()
        
        best_curve_HQ=best_curve[self.get_HQ_index()]
        x_current_HQ=self.x[self.get_HQ_index()]
        x_current_no_HQ=self.x[~self.get_HQ_index()]
        data_HQ = self.y[self.get_HQ_index()]
        data_no_HQ = self.y[~self.get_HQ_index()]
        freq_med_HQ = freq_med[self.get_HQ_index()]
        x_current=self.x 
        data = self.y 
        if x_compare_to_psd is not None:
            I_HQ = (x_current_HQ >=np.min(x_compare_to_psd)) & (x_current_HQ <=np.max(x_compare_to_psd))
          
            
            data_HQ=data_HQ[I_HQ]
            
            best_curve_HQ=best_curve_HQ[I_HQ]
            x_current_HQ=x_current_HQ[I_HQ]
            freq_med_HQ  = np.interp(x_current_HQ, x_compare_to_psd, psd_compare_to_psd)
            
            I = (self.x >=np.min(x_compare_to_psd)) & (self.x <=np.max(x_compare_to_psd))
            data=data[I]
            best_curve= best_curve[I] # take only wahts relevant             
            freq_med= psd_compare_to_psd
            x_current= x_compare_to_psd    
            
        
        plt.figure(figsize=(20,10))
        ax1= plt.subplot(1,1,1)
        if log:
            plt.loglog(x_current,data,label='data',color='k',alpha=0.7)
            plt.loglog(x_current_no_HQ,data_no_HQ,'or',label='Low Quality data',alpha=0.6)
            plt.loglog(x_current,best_curve,'-c',label='Max likelihood')
            plt.loglog(x_current,freq_med,'indigo',label= 'LIGO asd')
            
            
            plt.ylabel('Amplitude spectral density $[1/\\sqrt{\mathrm{Hz}}]$',fontsize=14)
            plt.xlabel('freq. [Hz]',fontsize=14)
             
            plt.legend(framealpha=0.0,frameon=False, ncol=1,loc="upper center",fontsize=14)
            #plt.grid(True,linestyle='--')
            plt.grid(False)
            
            ax1_1 = ax1.twinx()
            ax1_1.semilogx(x_current,np.cumsum(self._log_l_arr(data,best_curve))-np.cumsum(self._log_l_arr(data,freq_med)),'-k',label = ' $\\Delta \\sum{ \\log \\mathcal{L}_{All Data}}$')
            ax1_1.semilogx(x_current_HQ,np.cumsum(self._log_l_arr(data_HQ,best_curve_HQ))-
                           np.cumsum(self._log_l_arr(data_HQ,freq_med_HQ)),'-m',alpha=0.5,label=' $\\Delta \\sum{ \\log \\mathcal{L}_{High Quality Data}}$')
            
            plt.ylabel('$\\Delta \\sum{ \\log \\mathcal{L}}$',fontsize=14)
          
             
            plt.legend(framealpha=0.0,frameon=False, ncol=1,loc="lower right",fontsize=14)
            #plt.grid(True,linestyle='--')
                
            plt.grid(False)
            ax1.grid(False)
            ax1_1.grid(False)
            plt.savefig(self.sampling_label+'_'+ self.det +'_'  +'tbilby_LIGO_compare_logLikelihood.pdf',format='pdf',dpi= 250,bbox_inches="tight")

            
            
            
        else:    
            plt.plot(self.x,self.y,label='data')
            plt.plot(self.x,best_curve)
            ax1_1 = ax1.twinx()
            ax1_1.plot(self.x,np.cumsum(self._log_l_arr(self.y,best_curve))-np.cumsum(self._log_l_arr(self.y,freq_med)),'-k',label = 'LQ')
            ax1_1.plot(self.x[self.get_HQ_index()],np.cumsum(self._log_l_arr(self.y[self.get_HQ_index()],best_curve[self.get_HQ_index()]))-
                           np.cumsum(self._log_l_arr(self.y[self.get_HQ_index()],freq_med[self.get_HQ_index()])),'-m',alpha=0.5,label='HQ')
            
        
        plt.grid(False)
        ax1.grid(False)
        ax1_1.grid(False)
        plt.legend()
        
    def plot_maxL_at_trigger(self,alternative_trigger):
        
        best_curve= self.get_max_likelihood_curve()
        x_med, freq_med = self._get_GW_data_meidan()
        
        
        plt.figure()
        ax1= plt.subplot(1,1,1)
        x,y = self._get_GW_data_by_trigger(alternative_trigger, psd_duration=self.duration)
        plt.loglog(x,y,label='data')
        ax1_1 = ax1.twinx()
        ax1_1.semilogx(x,np.cumsum(self._log_l_arr(y,best_curve))-np.cumsum(self._log_l_arr(y,freq_med)),'-k',label = 'LQ')
        ax1_1.semilogx(x[self.get_HQ_index()],np.cumsum(self._log_l_arr(y[self.get_HQ_index()],best_curve[self.get_HQ_index()]))-
                       np.cumsum(self._log_l_arr(y[self.get_HQ_index()],freq_med[self.get_HQ_index()])),'-m',alpha=0.5,label='HQ')
        plt.grid(False)
        ax1.grid(False)
        ax1_1.grid(False)
        plt.title('at trigger = ' + str(alternative_trigger))
        plt.legend()
        
        
    def test_estimate_log_likelihood(self,start_time,test_time_sec= 60,plot_if_negative=False ):
            best_curve= self.get_max_likelihood_curve()
            x_med, freq_med = self._get_GW_data_meidan()            
            self._log_l_dist(self.x,median_arr=freq_med,fit_arr=best_curve,I_keep_bins=self.get_HQ_index(),start_time=start_time,test_secs=test_time_sec,plot_if_negative=plot_if_negative )
            
    def mixture_model_estimate_log_likelihood(self,max_likelihood_curve_0,max_likelihood_curve_1,
                                              I_keep_bins0,I_keep_bins1,start_time,lambda0=0.5,test_time_sec= 60 ):     
    
        x_med, freq_med = self._get_GW_data_meidan()
            
        self._log_l_dist_mixture(self.x,median_arr=freq_med,fit_arr0=max_likelihood_curve_0,fit_arr1=max_likelihood_curve_1,
                                 I_keep_bins0=I_keep_bins0,I_keep_bins1=I_keep_bins1,start_time=start_time,lambda0=lambda0,test_secs=test_time_sec )    
        
    def generate_posterior_samples(self,number_of_samples,genrate_mask=True):
            
                  
            self._merge_and_generate_samples(file_list = self.result_fnames,fname_full_range=self.fname_full_range,number_of_samples=number_of_samples,x=self.x,genrate_mask=genrate_mask)
            
            print('Done generating posterior_samples')
        
    
    
    def _load_values_from_pickle(self,filename):
         with open(filename, 'rb') as f:
             values = pickle.load(f)
             
         return np.stack(values)
     


    
    def _get_psd_from_ligo(self,f):    
        # = f['C01:IMRPhenomXPHM']['psds']
        ret_dict={}
        psd_data= f['C01:IMRPhenomXPHM']['psds']
        for ifo_name in list(psd_data):
        
            psd_freq_array = psd_data[ifo_name][:,0]
            I = (psd_freq_array > self.minimum_frequency) &  (psd_freq_array < self.maximum_frequency)
            frq = psd_data[ifo_name][I,0]
            vals = f['C01:IMRPhenomXPHM']['psds'][self.det][I,1]
            ret_dict[ifo_name] = np.vstack(list([frq,vals]))
    
       
        return ret_dict

    
    
    def Validate_sampled_samples(self,PE_file=None):
        
        
        fname= self.sampling_label+'asd_samples_'+self.det+'.pkl'
        print('Loading ' + fname)
        asds = self._load_values_from_pickle(fname)
        x,y = self._get_GW_data()
        x_med,y_med = self._get_GW_data_meidan()
        
        

        
        if PE_file is not None:
            import h5py
            f = h5py.File(PE_file, 'r')
            PE_psds_dict = self._get_psd_from_ligo(f)
       

        plt.figure(figsize=(20,10))
        plt.loglog(x,y,'k')
        plt.loglog(x_med,y_med,alpha=0.8,color='r',label='median ASD')
        for k in np.arange(10):
            tmp_lab = '_Hidden label'
            if k==0:
                tmp_lab='Posterior Sampled ASD'
            plt.loglog(x,asds[k,:],alpha=0.4,color='c', label=tmp_lab)
        if PE_file is not None:    
            plt.loglog(PE_psds_dict[self.det][0],np.sqrt(PE_psds_dict[self.det][1]),color='indigo',label='LIGO ASD',alpha=0.7)    

        
        plt.ylabel('Amplitude spectral density $[1/\\sqrt{\mathrm{Hz}}]$',fontsize=12)
        plt.xlabel('freq. [Hz]',fontsize=12)
         
        plt.legend(framealpha=0.0,frameon=False, ncol=1,loc="upper right",fontsize=12)
        plt.grid(True,linestyle='--')
        plt.savefig(self.sampling_label+'_'+ self.det +'_'  +'tbilby_psd_sampling.pdf',format='pdf',dpi= 1200,bbox_inches="tight")

        plt.title(self.sampling_label+': ' + self.det)
        
        
        if PE_file is not None:
            
            self.estimate_log_likelihood(log=True,x_compare_to_psd=PE_psds_dict[self.det][0],psd_compare_to_psd=np.sqrt(PE_psds_dict[self.det][1]))
        
        
        
        #plt.legend()
        
        
    def RunDefaultValidations(self,plot_if_negative=False):
        self.plot_maxL()
        self.plot_estimate_normal_goodness_of_fit()
        self.estimate_log_likelihood()
        self.test_estimate_log_likelihood(self.end_time+5,test_time_sec= 60,plot_if_negative=plot_if_negative)    


