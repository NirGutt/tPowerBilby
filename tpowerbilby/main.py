import bilby 
import tbilby
from gwpy.timeseries import TimeSeries
from scipy import stats

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
import matplotlib.pyplot as plt
import sys
from bilby.core.prior import ConditionalLogUniform, LogUniform
import asd_utilies 
import asd_data_manipulations 
import gwpy

import json
# Check if the correct number of command-line arguments are provided

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: didnt find a config file:  python my_script.py ASD_config.json")
        sys.exit(1)
else:
     print("Usage: didnt find a config file:  python my_script.py ASD_config.json")
     sys.exit(1)



# Get the path to the configuration file from the command-line arguments
config_file_path = sys.argv[1]
with open(config_file_path, 'r') as file:
    config = json.load(file)



detectors = ["H1", "L1","V1"]

config =asd_utilies.handle_config(config)

# settings: 
use_simpler_lorenztain_in_round_0=config['use_simpler_lorenztain_in_round_0']
det=config['det']
if det not in detectors:
    print('mmm, not sure how break this to you, but you provided me with a funny detector name. See you later mate!')
    sys.exit()
    


split_run = config['split_run']
user_label =config['user_label']
trigger_time = config['trigger_time']
maximum_frequency = config['maximum_frequency']
minimum_frequency = config['minimum_frequency']
roll_off = config['roll_off']  # Roll off duration of tukey window in seconds, default is 0.4s
duration = config['duration']  # Analysis segment duration
post_trigger_duration = config['post_trigger_duration']  # Time between trigger time and end of segment


# these are a by product of the settings 
label = 'tBilby_ASD_'+user_label+'_'+det 
outdir='Runs/'+label+'/'+config['outdir']
asd_utilies.check_folder_and_open(outdir)
number_of_rounds=2 
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration
psd_duration = duration




# these two function uses tbilby so we keep them here 
def get_max_likelihood_params_list(section_num,round_num,freq_split_vec,label,outdir):
    import os 
    # load result file 
    max_likeli_list=[]
    for section in np.arange(1,len(freq_split_vec)):
        if section ==section_num: # skip the current section 
            continue
        
        # select the latest one 
        label_sampling_result=''
        for r in np.arange(round_num+1):
            label_sampling_result_tmp=outdir+'/'+label+'_round_'+str(r)+'_'+str(freq_split_vec[section-1])+'_'+str(freq_split_vec[section])+'_result.json'
            if os.path.exists(label_sampling_result_tmp): 
                label_sampling_result=label_sampling_result_tmp
            
       
        if len(label_sampling_result) > 2: # found something 
            #label_sampling_result=outdir+'/'+label+'_'+str(freq_split_vec[section-1])+'_'+str(freq_split_vec[section])+'_result.json'
            result = bilby.result.read_in_result(filename=label_sampling_result)    
            # keep the highest Z model we got.
            result,cols = tbilby.core.base.preprocess_results(result,componant_functions_dict,remove_ghost_samples=False,return_samples_of_most_freq_component_function=True)
            best_params_post = tbilby.core.base.extract_maximal_likelihood_param_values(result, model=model)
            max_likeli_list.append(best_params_post)
            
    return max_likeli_list    


def get_smooth_function_dict(f_init,f_final,label,outdir):
    
    full_range_str='_full_range_result.json'
    label_sampling_result= outdir+'/'+label+'_round_0_'+str(f_init)+'_'+str(f_final)+full_range_str
    print('Getting results for broad band from:')
    print(label_sampling_result)
    result = bilby.result.read_in_result(filename=label_sampling_result)   
    result,cols = tbilby.core.base.preprocess_results(result,componant_functions_dict,remove_ghost_samples=False,return_samples_of_most_freq_component_function=True)
    best_params_post = tbilby.core.base.extract_maximal_likelihood_param_values(result, model=model)
    # keep the power law params 
    out_dict={}
    out_dict['n_exp']=int(best_params_post['n_exp'])
    for n in np.arange(n_exp):
        out_dict['A'+str(int(n))]=best_params_post['A'+str(int(n))]
        out_dict['lamda'+str(int(n))]=best_params_post['lamda'+str(int(n))]
    return out_dict     




class TransdimensionalConditionalBeta_Amp(tbilby.core.prior.TransdimensionalConditionalBeta):
    #self.__class__.__name__ = 'myTransdimensionalConditionalUniform'
    #self.__class__.__qualname__ = 'myTransdimensionalConditionalUniform'
    def set_extra_data(self,x,y):
        self.y=y
        self.x=x 
        # build interpolation map x-> min, x-> max       
        y_min = asd_utilies.min_rolling(y,20)      
        self.fdata = interpolate.interp1d(x, 1.1*self.y,fill_value="extrapolate") # allow for 10% extra 
        self.fmin = interpolate.interp1d(x[10:-9], y_min,fill_value="extrapolate")                               
        self.Nsigma = 2 # for now let be conservative 
    
               
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            componant_function_number= len(self.Amp)
              
            #extract its location 
            x_val = self.loc[componant_function_number]
        
            if isinstance(x_val, float):
                x_val=[x_val]
                
            minimum = self.fmin(x_val) # set the minimum to be the location of the last peak             
            maximum_data = self.fdata(x_val)
            
            est_smooth_func=np.zeros(np.array(x_val).shape)
            for n in np.arange(np.max(np.max(self.n_exp)).astype(int)):
                est_smooth_func += (self.n_exp > n).astype(float)*exp(x_val,self.A[n],self.lamda[n])   
                        
            new_minimum = est_smooth_func * self.Nsigma
            #minimum = new_minimum # wrong 
            minimum = est_smooth_func * (self.Nsigma-1) # at least the number of sigmas from the median 
            # these settaltis makes a difference when teh peak is realtively small SNR~3 - ish             
            maximum  = maximum_data - est_smooth_func
            
            if isinstance(new_minimum,np.ndarray):
                too_big_in_japan =  new_minimum > maximum_data
                minimum[too_big_in_japan] = 0.9
                maximum[too_big_in_japan] = 1
            else: # float 
               
                if isinstance(new_minimum,(np.floating,float)):
                    if  new_minimum > maximum_data:
                        maximum = np.array([1])
                        minimum = np.array([0.9])
             
            
            # check if we are over the maximum , if so, this means this is not a 
                        
            
            if len(minimum)==1:
                minimum=minimum[0]
                maximum=maximum[0]

                
            return dict(minimum=minimum,maximum=maximum)
        
        
class TransdimensionalConditionalTransInterped_loc(tbilby.core.prior.TransdimensionalConditionalTransInterped):
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            trans_min = self.minimum
            trans_max = self.maximum
    
            return dict(trans_min=trans_min,trans_max=trans_max) 
            
    
class TransdimensionalConditionalUniform_lamda(tbilby.core.prior.TransdimensionalConditionalUniform):   
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            minimum = self.minimum
            if(len(self.lamda)>0): # handle the first mu case
                minimum = self.lamda[-1] # set the minimum to be the location of the last peak 
                           
            return dict(minimum=minimum)


class TransdimensionalConditionalUniform_sAmp(tbilby.core.prior.TransdimensionalConditionalUniform):
    def set_extra_data(self,x,y):
        self.y=y
        self.x=x 
        # build interpolation map x-> min, x-> max 
       
        y_min = asd_utilies.min_rolling(y,20)        
        sigmas=y[50:-50]/asd_utilies.med_rolling(y,101)
        I = sigmas < 3.5
        x_mod = x[50:-50][I]        
        y_max = asd_utilies.max_rolling(y[50:-50][I],21)        
        
        x_for_fmax = x_mod[10:-10]
        y_for_fmax = y_max
        
        x_for_fmax = np.append(x[0], x_for_fmax)
        y_for_fmax = np.append(np.max(y[:60]), y_for_fmax)
        
        
        x_for_fmax = np.append(x_for_fmax,x[-1])
        y_for_fmax = np.append(y_for_fmax,np.max(y[-60:]))
        
        
        self.fmax = interpolate.interp1d(x_for_fmax,y_for_fmax,fill_value="extrapolate")
        self.fmin = interpolate.interp1d(x[10:-9], y_min,fill_value="extrapolate")            




class TransdimensionalConditionalUniform_s0Amp(TransdimensionalConditionalUniform_sAmp):    
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value                     
            #extract its location 
            x_val = self.s0x_center        
            if isinstance(x_val, float):
                x_val=[x_val]                
            minimum = np.zeros(np.array(x_val).shape)
            maximum = self.fmax(x_val)    # probably should be zero                           
            if len(minimum)==1:
                minimum=minimum[0]
                maximum=maximum[0]
            return dict(minimum=minimum,maximum=maximum)       

class TransdimensionalConditionalUniform_s1Amp(TransdimensionalConditionalUniform_sAmp):
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value                     
            #extract its location 
            x_val = self.s1x_center        
            if isinstance(x_val, float):
                x_val=[x_val]                
            minimum = np.zeros(np.array(x_val).shape) 
            maximum = self.fmax(x_val)    # probably should be zero                           
            if len(minimum)==1:
                minimum=minimum[0]
                maximum=maximum[0]
            return dict(minimum=minimum,maximum=maximum)       

class TransdimensionalConditionalUniform_s2Amp(TransdimensionalConditionalUniform_sAmp): 
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value                     
            #extract its location 
            x_val = self.s2x_center        
            if isinstance(x_val, float):
                x_val=[x_val]                
            minimum = np.zeros(np.array(x_val).shape) # set the minimum to be the location of the last peak 
            maximum = self.fmax(x_val)    # probably should be zero                           
            if len(minimum)==1:
                minimum=minimum[0]
                maximum=maximum[0]
            return dict(minimum=minimum,maximum=maximum)       

class TransdimensionalConditionalUniform_s3Amp(TransdimensionalConditionalUniform_sAmp):
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value                     
            #extract its location 
            x_val = self.s3x_center        
            if isinstance(x_val, float):
                x_val=[x_val]                
            minimum = np.zeros(np.array(x_val).shape) # probably should be zero                           
            maximum = self.fmax(x_val)    
            if len(minimum)==1:
                minimum=minimum[0]
                maximum=maximum[0]
            return dict(minimum=minimum,maximum=maximum)       



def exp(x,A,lamda):  
  xfunc = x.copy() # start from zero offset 
  return A*np.power(xfunc,lamda)

def improved_lorentzian(x, loc, Amp, gam,zeta,tau):
    # damped tails lorentzian, after 3sigma start dying out 
  
    damped_alpha = np.zeros(x.shape) 
    df_i = zeta *gam 
    I_left_tail = x < loc - df_i
    I_right_tail =x > loc + df_i
    
    damped_alpha [I_right_tail | I_left_tail] = tau 

    return np.exp(-damped_alpha*(np.abs(x-(loc))-df_i)) *Amp * gam**2 / ( gam**2 + ( x - loc )**2)

def lorentzian(x, loc, Amp, gam):
    # damped tails lorentzian, after 3sigma start dying out 
    decay =1
    damped_alpha = np.zeros(x.shape) 
    df_i = loc/50
    I_left_tail = x < loc - df_i
    I_right_tail =x > loc + df_i
    #damped_alpha[I_left_tail] = decay
    damped_alpha [I_right_tail | I_left_tail] = decay 
    return np.exp(-damped_alpha*(x-df_i)/(df_i)) *Amp * gam**2 / ( gam**2 + ( x - loc )**2)


def shaplet_func(x_in, deg =0 ,beta=1):
    x=x_in/beta
    coef= np.zeros((deg+1,))
    coef[deg]=1
      
    n = deg
    
    const =   1/np.sqrt(2**n * np.pi**(0.5) * np.math.factorial(n)) 
    weight = np.exp(-0.5*x**2) 
    poly = np.polynomial.hermite.Hermite(coef=coef.astype(int))(x)

    vals = const*poly*weight/np.sqrt(beta)
    #vals[vals<0]=0
    
    return vals 

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




class myGaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, function):
        
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
        #print(self.function_keys)
        if 'sigma' in self.function_keys: # remove sigma from function keys otherwise it get sent to the model
           self.function_keys.pop['sigma']
           print('Removing sigma')

      

    def log_likelihood(self):

        model_parameters = {k: self.parameters[k] for k in self.function_keys if k != 'sigma'}        
        est_asd = self.function(self.x, **model_parameters)      
        
        log_l =  np.sum(-self.normalization*(self.y/est_asd)**2 -np.log(est_asd**2))
        
        return log_l 

x,y= asd_data_manipulations.get_GW_data(config,start_time)


xx_main,yy_main = asd_data_manipulations.create_peaks_prior(x,y,config['use_low_freq_detector'],config['low_freq_limit_detector'] )


freq_split_vec = asd_data_manipulations.estimate_natural_parts(x,y,minimum_frequency,maximum_frequency,config['min_freq_spacing'],config['min_freq_segment'])



start_i,round_i = asd_utilies.check_from_where_to_start(freq_split_vec,number_of_rounds,label,outdir)
if start_i< 0:
    print('All result files are already avialble to you! dont make me sample this again :(, I am quitting ! ')
    sys.exit()
    
print('Starting from section number ' + str(start_i) + ' note the 1 is the first one, not zero ')    

if not split_run: # take the entire range 
    freq_split_vec = [freq_split_vec[0],freq_split_vec[-1]]    
    print('no need for rounds, setting one round  ')
    round_i=0
    number_of_rounds=1


for round_num in np.arange(round_i,number_of_rounds):
    if round_num> round_i:
        # we rae done with the intial round, we should set back start_i to 0.. 
        start_i=1

    for section in np.arange(start_i,len(freq_split_vec)+1):
        print('gettign down with round '+ str(round_num) + ' section ' + str(section))
        
        
        
        if section== len(freq_split_vec): # this the case when we finished the entire round and we are about to embark on the extra fit
            f_i=freq_split_vec[0]
            f_f=freq_split_vec[-1]
            
            
            if round_num==1:
                print('Mama raised no fool!, we are at the second round and we are out of sections. quitting mate !')
                sys.exit()
            
        else:
            f_i = freq_split_vec[section-1]
            f_f = freq_split_vec[section]
        
        
        x,y= asd_data_manipulations.get_GW_data(config=config,start_time=start_time,f_i=f_i,f_f=f_f)
    
    # take only the small part 
        
        Inx_xx = (xx_main>f_i) & (xx_main <f_f)
        xx=xx_main[Inx_xx]
        yy=yy_main[Inx_xx]
    
    
    
    
        n_exp=config['n_exp']
        n_peaks=config['n_lines']
        n_shaplets0=5
        n_shaplets1=5
        n_shaplets2=5
        n_shaplets3=5
    
    
        componant_functions_dict={}
        componant_functions_dict[exp]=(n_exp,'A','lamda')
        
        if use_simpler_lorenztain_in_round_0 and round_num==0:
            componant_functions_dict[lorentzian]=(n_peaks,'loc', 'Amp', 'gam')
        else:        
            componant_functions_dict[improved_lorentzian]=(n_peaks,'loc', 'Amp', 'gam','zeta','tau')
        componant_functions_dict[shaplets0]=(n_shaplets0,'s0Amp', True,'n_shaplets0')
        componant_functions_dict[shaplets1]=(n_shaplets1,'s1Amp', True,'n_shaplets1')
        componant_functions_dict[shaplets2]=(n_shaplets2,'s2Amp', True,'n_shaplets2')
        componant_functions_dict[shaplets3]=(n_shaplets3,'s3Amp', True,'n_shaplets3')
    
    
    
    
        model = tbilby.core.base.create_transdimensional_model('model',  componant_functions_dict,returns_polarization=False,SaveTofile=True)
    
    
        priors_t = bilby.core.prior.dict.ConditionalPriorDict()
    
        priors_t['n_exp'] = tbilby.core.prior.DiscreteUniform(1,n_exp,'n_exp')
        priors_t  = tbilby.core.base.create_plain_priors(LogUniform,'A',n_exp,prior_dict_to_add=priors_t,minimum=1e-30, maximum=1e-13)  
        priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_lamda,'lamda',nmax=n_exp,nested_conditional_transdimensional_params=['lamda'],conditional_params=[],prior_dict_to_add=priors_t,minimum=-10,maximum=2)
    
    
        
        # adding the prior accordin to the user choice 
        if use_simpler_lorenztain_in_round_0 and round_num==0:
            priors_t['n_lorentzian'] = tbilby.core.prior.DiscreteUniform(0,n_peaks,'n_lorentzian')            
        else:    
            priors_t['n_improved_lorentzian'] = tbilby.core.prior.DiscreteUniform(0,n_peaks,'n_improved_lorentzian')
            priors_t = tbilby.core.base.create_plain_priors(Uniform,'zeta',n_peaks,prior_dict_to_add=priors_t,minimum=0.1, maximum=5)
            priors_t = tbilby.core.base.create_plain_priors(Uniform,'tau',n_peaks,prior_dict_to_add=priors_t,minimum=0.5, maximum=10)
        
        priors_t = tbilby.core.base.create_plain_priors(LogUniform,'gam',n_peaks,prior_dict_to_add=priors_t,minimum=0.01, maximum=1)        
        priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalTransInterped_loc,'loc',nmax=n_peaks,nested_conditional_transdimensional_params=['loc'],conditional_params=[],prior_dict_to_add=priors_t,xx=xx,yy=yy,SaveConditionFunctionsToFile=False)
    
        priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalBeta_Amp,'Amp',nmax=n_peaks,nested_conditional_transdimensional_params=['Amp'],conditional_transdimensional_params=['loc',{'A':n_exp,'lamda':n_exp}],conditional_params=['n_exp'],SaveConditionFunctionsToFile=True,alpha = 4, beta=0.2,minimum=10,maximum=1000)
    
        for k in priors_t_temp.keys():
            priors_t_temp[k].set_extra_data(x,y)    
            priors_t[k] = priors_t_temp[k] # set the priors accordingly 
    
    
        priors_t['n_shaplets0'] = tbilby.core.prior.DiscreteUniform(0,n_shaplets0,'n_shaplets0')
        priors_t['n_shaplets1'] = tbilby.core.prior.DiscreteUniform(0,n_shaplets1,'n_shaplets1')
        priors_t['n_shaplets2'] = tbilby.core.prior.DiscreteUniform(0,n_shaplets2,'n_shaplets2')
        priors_t['n_shaplets3'] = tbilby.core.prior.DiscreteUniform(0,n_shaplets3,'n_shaplets3')
    
        max_arr = asd_utilies.max_rolling(y,10)
        min_arr = asd_utilies.min_rolling(y,10)
        Amp_max_shaplets = 10*np.max(max_arr-min_arr)
    
    
        priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_s0Amp,'s0Amp',nmax=n_shaplets0,nested_conditional_transdimensional_params=['s0Amp'],conditional_transdimensional_params={'A':n_exp,'lamda':n_exp},conditional_params=['n_exp','s0x_center'],SaveConditionFunctionsToFile=True,minimum=0,maximum=10**(-23))
        for k in priors_t_temp.keys():
            priors_t_temp[k].set_extra_data(x,y)    
            priors_t[k] = priors_t_temp[k]
        #priors_t  = tbilby.create_plain_priors(LogUniform,'s0Amp',n_shaplets0,prior_dict_to_add=priors_t,minimum=Amp_max_shaplets/10000, maximum=Amp_max_shaplets)      
        priors_t['s0beta'] = tbilby.core.prior.Uniform(minimum=0.5, maximum=5) # common to all shaplets 
        priors_t['s0x_center'] = tbilby.core.prior.Uniform(minimum=np.min(x), maximum=np.max(x)) # common to all shaplets 
    
    
    
        priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_s1Amp,'s1Amp',nmax=n_shaplets1,nested_conditional_transdimensional_params=['s1Amp'],conditional_transdimensional_params={'A':n_exp,'lamda':n_exp},conditional_params=['n_exp','s1x_center'],SaveConditionFunctionsToFile=True,minimum=0,maximum=10**(-23))
        for k in priors_t_temp.keys():
            priors_t_temp[k].set_extra_data(x,y)    
            priors_t[k] = priors_t_temp[k]
        #priors_t  = tbilby.create_plain_priors(LogUniform,'s1Amp',n_shaplets0,prior_dict_to_add=priors_t,minimum=Amp_max_shaplets/10000, maximum=Amp_max_shaplets)      
        priors_t['s1beta'] = tbilby.core.prior.Uniform(minimum=0.5, maximum=5) # common to all shaplets 
        priors_t['s1x_center'] = tbilby.core.prior.Uniform(minimum=np.min(x), maximum=np.max(x)) # common to all shaplets 
    
        priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_s2Amp,'s2Amp',nmax=n_shaplets2,nested_conditional_transdimensional_params=['s2Amp'],conditional_transdimensional_params={'A':n_exp,'lamda':n_exp},conditional_params=['n_exp','s2x_center'],SaveConditionFunctionsToFile=True,minimum=0,maximum=10**(-23))
        for k in priors_t_temp.keys():
            priors_t_temp[k].set_extra_data(x,y)    
            priors_t[k] = priors_t_temp[k]
        #priors_t  = tbilby.create_plain_priors(LogUniform,'s2Amp',n_shaplets0,prior_dict_to_add=priors_t,minimum=Amp_max_shaplets/10000, maximum=Amp_max_shaplets)      
        priors_t['s2beta'] = tbilby.core.prior.Uniform(minimum=0.5, maximum=5) # common to all shaplets 
        priors_t['s2x_center'] = tbilby.core.prior.Uniform(minimum=np.min(x), maximum=np.max(x)) # common to all shaplets 
    
        priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_s3Amp,'s3Amp',nmax=n_shaplets3,nested_conditional_transdimensional_params=['s3Amp'],conditional_transdimensional_params={'A':n_exp,'lamda':n_exp},conditional_params=['n_exp','s3x_center'],SaveConditionFunctionsToFile=True,minimum=0,maximum=10**(-23))
        for k in priors_t_temp.keys():
            priors_t_temp[k].set_extra_data(x,y)    
            priors_t[k] = priors_t_temp[k]
        #priors_t  = tbilby.create_plain_priors(LogUniform,'s3Amp',n_shaplets0,prior_dict_to_add=priors_t,minimum=Amp_max_shaplets/10000, maximum=Amp_max_shaplets)      
        priors_t['s3beta'] = tbilby.core.prior.Uniform(minimum=0.5, maximum=5) # common to all shaplets 
        priors_t['s3x_center'] = tbilby.core.prior.Uniform(minimum=np.min(x), maximum=np.max(x)) # common to all shaplets 
            
    
        
        max_likeli_list = get_max_likelihood_params_list(section,round_num,freq_split_vec,label,outdir)
        
        # retake the data including all previous parts , clean it from non smooth part 
        take_section = section
        # this used to be the method, but we found that there is a leakage between adjusent sections using the shaplets   
        #if round_num!=0 or 
        if section== len(freq_split_vec): # we have teh entire data already 
              take_section=-1  
               
        x,y= asd_data_manipulations.get_GW_data(config=config,start_time=start_time,f_i=freq_split_vec[0],f_f=freq_split_vec[take_section])

        # the model here coudl have both types of lorentzians, we take care of it inside the clean_data_from_dirty_dirty_points function  
        x_clean,y_clean = asd_data_manipulations.clean_data_from_dirty_dirty_points(x,y,max_likeli_list,model)
       
        
        if round_num!=0:
            print('Round number 1, setting the braod abnd to const.')
            power_law_dict = get_smooth_function_dict(freq_split_vec[0],freq_split_vec[-1],label=label,outdir=outdir)
            for name in power_law_dict.keys():
                priors_t[name]=power_law_dict[name]
                print(name)
            
        
    
        asd_utilies.check_data(x,y,x_clean,y_clean,freq_split_vec,section,round_num,outdir)
        
        full_range_str=''
        if section== len(freq_split_vec) and round_num==0:
            # remove the lines and shaplets from the fit 
            priors_t['n_shaplets0'] = 0
            priors_t['n_shaplets1'] = 0
            priors_t['n_shaplets2'] = 0
            priors_t['n_shaplets3'] = 0
            if use_simpler_lorenztain_in_round_0:
                priors_t['n_lorentzian'] =0
            else:
                priors_t['n_improved_lorentzian'] =0
            full_range_str='_full_range'
    
            label_sampling=label+'_round_'+str(round_num)+'_'+str(freq_split_vec[0])+'_'+str(freq_split_vec[-1])+full_range_str
        else:
            label_sampling=label+'_round_'+str(round_num)+'_'+str(freq_split_vec[section-1])+'_'+str(freq_split_vec[section])+full_range_str    
    
        likelihood = myGaussianLikelihood(x_clean,y_clean, model)
    
           
        
        
       
        print(label_sampling)
    

        result = bilby.run_sampler(
                likelihood=likelihood,
                priors=priors_t,
                #method='multi',
                #sample='rslice',
                sample='rwalk',
                nlive=700,
                walks=150,
                outdir=outdir,
                label=label_sampling,
                #clean=True,
                resume=True,               
                npool=16,
                )
    
    
