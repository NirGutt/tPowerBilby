from scipy import interpolate
import inspect
from bilby.core.prior import LogUniform
import bilby 
import tbilby
import re 
import numpy as np
import matplotlib.pyplot as plt
import os
from .asd_utilies import logger


class TransdimensionalConditionalLogUniform_Amp(tbilby.core.prior.TransdimensionalConditionalLogUniform): 
    
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            
            maximum = self.maximum
            minimum = self.minimum

            if not hasattr(self, "keep_delta"):
                self.keep_delta=np.log10(maximum) - np.log10(minimum)  
                logger(f"initially setting, self.keep_delta = {self.keep_delta}","TransdimensionalConditionalLogUniform_Amp")

    
            if(len(self.A)>0): # handle the first mu case
                maximum = self.A[-1] # set the maximum to be the location of the last peak 
                if self.keep_delta>0:
                    minimum = 10**(np.log10(maximum) - self.keep_delta) # keeping the probability alive, so no shrinkage happens 
                    #logger(f"maximum = {maximum}","TransdimensionalConditionalUniform_lamda")
            
                            
                #logger(f"self.keep_delta = {self.keep_delta}","TransdimensionalConditionalUniform_lamda")
            #logger(f"min: {minimum}","TransdimensionalConditionalLogUniform_Amp")
            #logger(f"max: {maximum}","TransdimensionalConditionalLogUniform_Amp")
            return dict(minimum=minimum,maximum =maximum)


const_lamda_dict={}

class TransdimensionalConditionalUniform_lamda(tbilby.core.prior.TransdimensionalConditionalUniform): 
    
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            minimum = self.minimum
            maximum = self.maximum

            if not hasattr(self, "keep_delta"):                
                self.keep_delta=maximum - minimum 
                logger(f"initially setting, self.keep_delta = {self.keep_delta}","TransdimensionalConditionalUniform_lamda")

            if(len(self.lamda)>0): # handle the first mu case
                minimum = self.lamda[-1] # set the minimum to be the location of the last peak 
                if self.keep_delta>0:
                    maximum =minimum +self.keep_delta # keeping the probability alive, so no shrinkage happens 
                    #logger(f"maximum = {maximum}","TransdimensionalConditionalUniform_lamda")
                if self.name in const_lamda_dict:
                    minimum = np.ones_like(minimum) * const_lamda_dict[self.name] if isinstance(minimum, (np.ndarray, list)) else const_lamda_dict[self.name]
                    maximum = minimum+0.0001 

                #logger(f"self.keep_delta = {self.keep_delta}","TransdimensionalConditionalUniform_lamda")
            #logger(f"min: {self.name} {minimum}" ,"TransdimensionalConditionalUniform_lamda")
            #logger(f"max: {self.name} {maximum}","TransdimensionalConditionalUniform_lamda")
            return dict(minimum=minimum,maximum =maximum)

class TransdimensionalConditionalUniform_sAmp(tbilby.core.prior.TransdimensionalConditionalUniform):
    def set_extra_data(self,try_cls):
      
        # build interpolation map x-> min, x-> max 
        x_Amp_Shaplets_p,y_Amp_Shaplets_p = try_cls.GetShaplets_Amp_prior()        
        self.fmax = interpolate.interp1d(x_Amp_Shaplets_p,(2/3.5)*y_Amp_Shaplets_p,fill_value="extrapolate")
        

class TransdimensionalConditionalUniform_s0Amp(TransdimensionalConditionalUniform_sAmp):    

    def find_loc_variations(self,str_to_find):
        pattern = re.compile(r"\d+")
        for attr in dir(self):  # Iterate over all attributes of the instance
            if str_to_find in attr:
                if pattern.search(attr):   # Check if 'loc' is in the attribute name
                    return getattr(self, attr)  # Add to dictionary with name and value
        
        return None

    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value                     
            #extract its location 
            x_val =  self.find_loc_variations('x_center')
           
            if isinstance(x_val, float):
                x_val=[x_val]                
            minimum = np.zeros(np.array(x_val).shape)
            
            maximum = self.fmax(x_val)                         
            if len(minimum)==1:
                minimum=minimum[0]
                maximum=maximum[0]
            return dict(minimum=minimum,maximum=maximum)     


def exp(x,A,lamda):  
  xfunc = x.copy() # start from zero offset 
  return A*np.power(xfunc,lamda)

def shaplet_func(x_in, deg =0 ,beta=1):
    x=x_in/beta
    coef= np.zeros((deg+1,))
    coef[deg]=1
      
    n = deg
    
    const =   1/np.sqrt(2**n * np.pi**(0.5) * np.math.factorial(n)) 
    weight = np.exp(-0.5*x**2) 
    poly = np.polynomial.hermite.Hermite(coef=coef.astype(int))(x)

    vals = const*poly*weight/np.sqrt(beta)
  
    
    return vals 


def shaplets(x,n_shaplets,sx_center,sbeta,sAmp):
   xfunc = x.copy()    
   return sAmp*shaplet_func(xfunc-sx_center,n_shaplets,sbeta)

def shaplets0(x,n_shaplets0,s0x_center,s0beta,s0Amp):
   xfunc = x.copy()
    
   return s0Amp*shaplet_func(xfunc-s0x_center,n_shaplets0,s0beta)

def shaplets1(x,n_shaplets1,s1x_center,s1beta,s1Amp):
   xfunc = x.copy()
    
   return s1Amp*shaplet_func(xfunc-s1x_center,n_shaplets1,s1beta)


def shaplets2(x,n_shaplets2,s2x_center,s2beta,s2Amp):
   xfunc = x.copy()
    
   return s2Amp*shaplet_func(xfunc-s2x_center,n_shaplets2,s2beta)


def shaplets3(x,n_shaplets3,s3x_center,s3beta,s3Amp):
   xfunc = x.copy()
    
   return s3Amp*shaplet_func(xfunc-s3x_center,n_shaplets3,s3beta)

  
def process_lines_prior(gap_threshold,x,x_peaks,y_peaks_in,config):
        
        y_peaks=y_peaks_in.copy()
       
        # loop over the regions, for make sure it is even  
        
        if len(config['lines_prior']) % 2 == 0:
            
            # for each pair teh relevant indices and set then to 1 
            for i in range(0, len(config['lines_prior']), 2):
                # find the indices 
                Itmp = (x_peaks > config['lines_prior'][i])&(x_peaks < config['lines_prior'][i+1])
                y_peaks[Itmp]=1 # set something different from 0 
                logger(f"lines prior user setting {config['lines_prior'][i]} Hz-> {config['lines_prior'][i+1]} Hz is applied",inspect.currentframe().f_code.co_name ) 
        else:
            logger('lines prior user setting isnt of even length, ignoring ',inspect.currentframe().f_code.co_name )    
       
        # Find the indices where y > 0 (non-zero regions)
        non_zero_indices = np.where(y_peaks > 0)[0]    

        # Fill small gaps by checking distances between non-zero indices
        for i in range(1, len(non_zero_indices)):
            if non_zero_indices[i] - non_zero_indices[i-1] <= gap_threshold:
                # Linear interpolation for small gaps
                start, end = non_zero_indices[i-1], non_zero_indices[i]
                y_peaks[start:end] = np.interp(x_peaks[start:end], [x_peaks[start], x_peaks[end]], 
                [y_peaks[start], y_peaks[end]])
        
        extended_arr = y_peaks.copy()
        lines_prior = y_peaks.copy()
        non_zero_indices = np.nonzero(y_peaks)[0]
        lines_prior = np.interp(x,x_peaks,lines_prior)

        # Extend the region of non-zero values by +/- 10 cells
        for idx in non_zero_indices:
            lower_bound = max(0, idx - 20)  # Make sure we don't go below index 0
            upper_bound = min(len(y_peaks), idx + 20 + 1)  # Make sure we don't exceed the array length
            extended_arr[lower_bound:upper_bound] = y_peaks[idx]  # Set 
        y_peaks = extended_arr.copy()        
        y_peaks_int = np.interp(x,x_peaks,y_peaks)
        return y_peaks_int,lines_prior.copy()



class myGaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, function):
        duration=4
        self.x = x
        self.y = y
        self.N = len(x)
        self.normalization = 0.5* (4/duration)
        self.function = function

        # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(function).args

        del parameters[0]
        super().__init__(parameters=dict.fromkeys(parameters))
        self.parameters = dict.fromkeys(parameters)

        self.function_keys = self.parameters.keys()
        
        if 'sigma' in self.function_keys: # remove sigma from function keys otherwise it get sent to the model
           self.function_keys.pop['sigma']
           

      

    def log_likelihood(self):
        
        model_parameters = {k: self.parameters[k] for k in self.function_keys if k != 'sigma'}        
        est_asd = self.function(self.x, **model_parameters)      
        
        log_l =  np.sum(-self.normalization*(self.y/est_asd)**2 -np.log(est_asd**2))
        #log_l =  np.sum(-self.normalization*(self.y/(est_asd/np.sqrt(n)))**2 -np.log((est_asd/np.sqrt(n))**2))
        #log_l =  np.sum(-self.normalization*(self.y-est_asd)**2/(self.y/np.sqrt(n))**2 )
        
        return log_l 



def run_PL_fit(x,y,x_est,welch_y,preprocess_cls,outdir,config,label='',num_of_samples=1000,n_live=400,resume=False,n_exp=5,debug=False):
     
    sh_deg=4
    n_shaplets0=sh_deg
    n_shaplets1=sh_deg
    n_shaplets2=sh_deg
    n_shaplets3=sh_deg
    
    componant_functions_dict={}
    componant_functions_dict[exp]=(n_exp,'A','lamda')
   
    
    componant_functions_dict[shaplets0]=(n_shaplets0,'s0Amp', True,'n_shaplets0')
    componant_functions_dict[shaplets1]=(n_shaplets1,'s1Amp', True,'n_shaplets1')
    componant_functions_dict[shaplets2]=(n_shaplets2,'s2Amp', True,'n_shaplets2')
    componant_functions_dict[shaplets3]=(n_shaplets3,'s3Amp', True,'n_shaplets3')
    
    model = tbilby.core.base.create_transdimensional_model('model',  componant_functions_dict,returns_polarization=False,SaveTofile=False)
    
    priors_t = bilby.core.prior.dict.ConditionalPriorDict()
    priors_t['n_exp'] = tbilby.core.prior.DiscreteUniform(1,n_exp,'n_exp')
    
    #priors_t  = tbilby.core.base.create_plain_priors(LogUniform,'A',n_exp,prior_dict_to_add=priors_t,minimum=1e-30, maximum=1e-13)  
    # experimental version 
    priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalLogUniform_Amp,'A',nmax=n_exp,nested_conditional_transdimensional_params=['A'],conditional_params=[],prior_dict_to_add=priors_t,minimum=1e-30, maximum=1e-13)
    

    priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_lamda,'lamda',nmax=n_exp,nested_conditional_transdimensional_params=['lamda'],conditional_params=[],prior_dict_to_add=priors_t,minimum=-10,maximum=2)
    # set the values to consnta according to the user config



    for alpha_i in np.arange(n_exp):
        if f"alpha{alpha_i}" in config:
            if config[f"alpha{alpha_i}"]> -100: # which is the defualt set value 
                const_lamda_dict[f"lamda{alpha_i}"] = config[f"alpha{alpha_i}"]
                logger('setting lamda'+str(alpha_i) + '=' + str(config[f"alpha{alpha_i}"]),inspect.currentframe().f_code.co_name)
    
    for sh_no in np.arange(sh_deg):
        priors_t[f'n_shaplets{sh_no}'] = tbilby.core.prior.DiscreteUniform(0,sh_deg,f'n_shaplets{sh_no}')
    
        priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_s0Amp,
                                                                                f's{sh_no}Amp',nmax=sh_deg,
                                                                                nested_conditional_transdimensional_params=[f's{sh_no}Amp'],
                                                                                conditional_transdimensional_params=[],
                                                                                conditional_params=[f's{sh_no}x_center'],
                                                                                SaveConditionFunctionsToFile=False,minimum=0,maximum=10**(-23))
        for k in priors_t_temp.keys():
            priors_t_temp[k].set_extra_data(preprocess_cls)    
            priors_t[k] = priors_t_temp[k]
                
        priors_t[f's{sh_no}beta'] = tbilby.core.prior.LogUniform(minimum=1, maximum=500) 
        priors_t[f's{sh_no}x_center'] = tbilby.core.prior.Uniform(minimum=0,maximum=1000) # common to all shaplets 
    
   
    
   
    likelihood = myGaussianLikelihood(x,y, model)
    label_sampling_local=label+'_First_Phase_PL'
        # for seom reason the resume doesn't work, lets override it 
    file_path = outdir+'/'+label_sampling_local+'_result.json'
    logger(('checking if results are in ', file_path, ' resume is' , resume),inspect.currentframe().f_code.co_name)
    if resume and os.path.exists(file_path):     
        logger('loadigng results from ' + file_path,inspect.currentframe().f_code.co_name)
        
        result = bilby.read_in_result(filename=file_path) 
    else:   
   

        result = bilby.run_sampler(
                    likelihood=likelihood,
                    priors=priors_t,                 
                    sample='rwalk',
                    nlive=n_live,
                    
                    outdir=outdir,
                    label=label_sampling_local,
                    clean=(not resume),
                    resume=resume,               
                    npool=16,
                    )
    
  
    #sample first 
    rand_samples_1st= result.posterior.sample(frac = 1).copy()
    psd_keeper=[]    
   
    for sample_i in np.arange(num_of_samples):    
        model_parameters_1st =rand_samples_1st.iloc[sample_i].to_dict()   
        del model_parameters_1st['log_likelihood']
        del model_parameters_1st['log_prior']  
        psd_keeper.append(model(x_est,**model_parameters_1st))     
        tmp_arr= model(x_est,**model_parameters_1st)
       

    res_smooth,cols = tbilby.core.base.preprocess_results(result,componant_functions_dict,remove_ghost_samples=False,return_samples_of_most_freq_component_function=True)
    
    params= tbilby.core.base.extract_maximal_likelihood_param_values(result=res_smooth)
    
    del params['index']
    del params['log_likelihood']
    del params['log_prior']
 
    if debug:
        plt.figure()                            
        plt.loglog(x,y,alpha=0.4)        
        plt.loglog(x,welch_y,alpha=0.5)
        plt.loglog(x_est,model(x_est,**params),'-k')    
        plt.savefig(outdir+'/Validations/'+label_sampling_local+'.png')


    

   

   
    return psd_keeper,model(x_est,**params),model,rand_samples_1st
