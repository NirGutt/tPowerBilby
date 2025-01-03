import bilby 
from context import tbilby
import pickle
import numpy as np
from scipy import interpolate
import inspect
import sys
from bilby.core.prior import LogUniform, TruncatedGaussian

import Utils.asd_utilies as asd_utilies
import Utils.asd_data_manipulations as asd_data_manipulations 
import Utils.PreProcessingtPowerBilby as PreProcessingtPowerBilby
import Utils.PostProcessingtPowerBilby as PostProcessingtPowerBilby
import Utils.FirstPhase as FPH

import gwpy
import re 
import json
import concurrent.futures
import os

#some helper functions 
class TransdimensionalConditionalBeta_Amp(tbilby.core.prior.TransdimensionalConditionalLogUniform):
    
    
    def find_loc_variations(self,str_to_find):
        pattern = re.compile(r"\d+")
        for attr in dir(self):  # Iterate over all attributes of the instance
            if str_to_find in attr:
                if pattern.search(attr):   # Check if 'loc' is in the attribute name
                    return getattr(self, attr)  # Add to dictionary with name and value
        
        return None
    
    def set_extra_data(self):

        x_Amp_lines_p,y_Amp_lines_p_max,y_Amp_lines_p_min = PreProcesstPowerBilby_class.GetLines_Amp_prior()        
        # build interpolation map x-> min, x-> max               
        self.Amp_max_interp = interpolate.interp1d(x_Amp_lines_p,1.1*y_Amp_lines_p_max,fill_value="extrapolate") # allow for 10% extra                                    
        self.Amp_min_interp = interpolate.interp1d(x_Amp_lines_p,y_Amp_lines_p_min,fill_value="extrapolate")
        
    
               
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value

            componant_function_number= len(self.Amp)
                                
            #extract its location    
            x_val = self.loc[componant_function_number]                          
            if isinstance(x_val, float):
                x_val=[x_val]
                                    
            maximum = self.Amp_max_interp(x_val)
            minimum =self.Amp_min_interp(x_val)
                                     
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
    def set_extra_data(self):
      
        # build interpolation map x-> min, x-> max 
        x_Amp_Shaplets_p,y_Amp_Shaplets_p = PreProcesstPowerBilby_class.GetShaplets_Amp_prior()                 
        self.fmax = interpolate.interp1d(x_Amp_Shaplets_p,(2/3.5)*y_Amp_Shaplets_p,fill_value="extrapolate")
        

class TransdimensionalConditionalUniform_s0Amp(TransdimensionalConditionalUniform_sAmp):    

    def find_loc_variations(self,str_to_find):
        pattern = re.compile(r"\d+")
        for attr in dir(self):  # Iterate over all attributes of the instance
            if str_to_find in attr:
                if pattern.search(attr):   # Check if 'loc' is in the attribute name
                    return getattr(self, attr)  # Add to dictionary with name and value
        
        return []

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

class TransdimensionalConditionalUniform_s1Amp(TransdimensionalConditionalUniform_sAmp):
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value                     
            #extract its location 
            x_val = self.s1x_center        
            if isinstance(x_val, float):
                x_val=[x_val]                
            minimum = np.zeros(np.array(x_val).shape) 
            maximum = self.fmax(x_val)                          
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
            maximum = self.fmax(x_val)                              
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
            minimum = np.zeros(np.array(x_val).shape)                        
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

def shaplets4(x,n_shaplets4,s4x_center,s4beta,s4Amp):
   xfunc = x.copy()
    
   return s4Amp*shaplet_func(xfunc-s4x_center,n_shaplets4,s4beta)

def run_fit_on_segment(segmment_i,resume=False):

    x_clean,y_clean= asd_data_manipulations.get_GW_data(config,start_time)
    x_est=x_clean.copy()
    frq_bins_to_fit = I_keep_data_indices & (~I_FPH)
    x_clean=x_clean[frq_bins_to_fit] # fit only the the HQ data and everything that wasn't fittign in the first pahse 
    y_clean=y_clean[frq_bins_to_fit]
    f_i = PreProcesstPowerBilby_class.GetFrequency_SplitPoints()[segmment_i]
    f_f = PreProcesstPowerBilby_class.GetFrequency_SplitPoints()[segmment_i+1]
        
    # keep only the segment part     
    I_keep_local = (x_clean>=f_i) & (x_clean<=f_f) 
    x_clean=x_clean[I_keep_local]
    y_clean=y_clean[I_keep_local]
    local_no_lines_curve = no_lines_curve[frq_bins_to_fit]
    local_no_lines_curve = local_no_lines_curve[I_keep_local]
    
    # plot the data out
    round_num=1    
    asd_utilies.check_data(x,y,x_clean,y_clean,PreProcesstPowerBilby_class.GetFrequency_SplitPoints(),segmment_i,round_num,outdir+'/Validations')

    
    n_lines_per_split[segmment_i]=config['n_lines']
    n_peaks=config['n_lines']
    sh_deg=5
    n_shaplets_per_region =4

    Inx_xx = (xx_main>f_i) & (xx_main <f_f)
    
    # build the model 
    componant_functions_dict={}
    componant_functions_dict[improved_lorentzian]=(n_peaks,'loc', 'Amp', 'gam','zeta')
    componant_functions_dict[shaplets0]=(sh_deg,'s0Amp', True,'n_shaplets0')
    componant_functions_dict[shaplets1]=(sh_deg,'s1Amp', True,'n_shaplets1')
    componant_functions_dict[shaplets2]=(sh_deg,'s2Amp', True,'n_shaplets2')
    componant_functions_dict[shaplets3]=(sh_deg,'s3Amp', True,'n_shaplets3')
    
       
        
    model = tbilby.core.base.create_transdimensional_model('model',  componant_functions_dict,returns_polarization=False,SaveTofile=False)

    # define prriros 
    priors_t = bilby.core.prior.dict.ConditionalPriorDict()
    priors_t[f'n_improved_lorentzian'] = tbilby.core.prior.DiscreteUniform(0,n_lines_per_split[segmment_i],f'n_improved_lorentzian') 

           
    priors_t = tbilby.core.base.create_plain_priors(LogUniform,'gam',n_lines_per_split[segmment_i],prior_dict_to_add=priors_t,minimum=10**(-3), maximum=10)        
            
    xx=xx_main[Inx_xx]
    yy=yy_main[Inx_xx]
    print(('region',segmment_i,f_i,f_f))        
    print(('max xx', np.max(xx)))
    print(('min xx', np.min(xx)))

    priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalTransInterped_loc,f'loc',nmax=n_lines_per_split[segmment_i],nested_conditional_transdimensional_params=[],conditional_params=[],prior_dict_to_add=priors_t,xx=xx.copy(),yy=yy.copy(),
                                                                       minimum=PreProcesstPowerBilby_class.GetFrequency_SplitPoints()[segmment_i],
                                                                       maximum=PreProcesstPowerBilby_class.GetFrequency_SplitPoints()[segmment_i+1], 
                                                                       SaveConditionFunctionsToFile=False)
            
    priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalBeta_Amp,f'Amp',nmax=n_lines_per_split[segmment_i],
                                                                            nested_conditional_transdimensional_params=[f'Amp'],
                                                                            conditional_transdimensional_params=[f'loc'],
                                                                            conditional_params=[],
                                                                            SaveConditionFunctionsToFile=False,minimum=10,maximum=1000)
    
    for k in priors_t_temp.keys():
        priors_t_temp[k].set_extra_data()    
        priors_t[k] = priors_t_temp[k] # set the priors accordingly 
        

    priors_t = tbilby.core.base.create_plain_priors(TruncatedGaussian,f'zeta',n_lines_per_split[segmment_i],prior_dict_to_add=priors_t,mu=2.7,sigma=1.1,minimum=0.1, maximum=5)

            # set it to constant 
    priors_t[f'tau']=5.2

                
    for sh in np.arange(n_shaplets_per_region):

        priors_t[f'n_shaplets{sh}'] = tbilby.core.prior.DiscreteUniform(0,sh_deg,f'n_shaplets{sh}')
        
               
        priors_t_temp = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_s0Amp,
                                                                                f's{sh}Amp',nmax=sh_deg,
                                                                                nested_conditional_transdimensional_params=[f's{sh}Amp'],
                                                                                conditional_transdimensional_params=[],
                                                                                conditional_params=[f's{sh}x_center'],
                                                                                SaveConditionFunctionsToFile=True,minimum=0,maximum=10**(-23))
        for k in priors_t_temp.keys():
            priors_t_temp[k].set_extra_data()    
            priors_t[k] = priors_t_temp[k]
                
        priors_t[f's{sh}beta'] = tbilby.core.prior.LogUniform(minimum=0.5, maximum=10) #infcrease from 5Hz to 10Hz, common to all shaplets 
        priors_t[f's{sh}x_center'] = bilby.core.prior.interpolated.Interped(xx= xx_main.copy(), yy=yy_main.copy(), 
                                                                            minimum=PreProcesstPowerBilby_class.GetFrequency_SplitPoints()[segmment_i], 
                                                                            maximum=PreProcesstPowerBilby_class.GetFrequency_SplitPoints()[segmment_i+1])
               
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
      
        def lines_seperator_model(self):
            model_parameters = {k: self.parameters[k] for k in self.function_keys if k != 'sigma'}    
        
            est_asd_nolines = local_no_lines_curve.copy() # this gets updated from the first phase 
            Lines_only_asd = self.function(self.x, **model_parameters)           
            # max
            new_curve = np.max([Lines_only_asd,est_asd_nolines],axis=0)
          
            return new_curve  

        def log_likelihood(self):

            est_asd = self.lines_seperator_model()  
              
            log_l =  np.sum(-self.normalization*(self.y/est_asd)**2 -np.log(est_asd**2))
            
            return log_l 

    label_sampling_local=label+f'_Phase_Two_{f_i}_{f_f}'
    likelihood = myGaussianLikelihood(x_clean,y_clean, model)
    
    
    # for seom reason the resume doesn't work, lets override it 
    file_path = outdir+'/'+label_sampling_local+'_result.json'
    print(('checking if results are in ', file_path, ' resume is' , resume))
    if resume and os.path.exists(file_path):     
        print('loadigng results')
        result = bilby.read_in_result(filename=file_path) 
    else:    
        print('rerunning sampling')
        result = bilby.run_sampler(
                likelihood=likelihood,
                priors=priors_t,
                sample='rwalk',
                nlive=nlive,
                outdir=outdir,
                label=label_sampling_local,
                resume=resume,               
                npool=16,
                )
        
    # do the analysis here, otherwise it's a nightmare 
    # extract the maximum likleihood value
    # generate n_samples for later usage 

    rand_samples_2nd= result.posterior.sample(frac = 1)
    psd_keeper=[]    

    x_est = get_x_est(x_est)

    for sample_i in np.arange(N_samples):    
        model_parameters_2nd =rand_samples_2nd.iloc[sample_i].to_dict()      
        del model_parameters_2nd['log_likelihood']
        del model_parameters_2nd['log_prior']  

        Lines_only_asd = model(x_est,**model_parameters_2nd)
     
        final_curve_lines = np.max([Lines_only_asd,no_lines_curve.copy()],axis=0)
      
        psd_keeper.append(final_curve_lines)     
    
    res_lines,cols = tbilby.core.base.preprocess_results(result,componant_functions_dict,remove_ghost_samples=False,return_samples_of_most_freq_component_function=True)
    
    params_ph2= tbilby.core.base.extract_maximal_likelihood_param_values(result=res_lines)
    log_prior = params_ph2['log_prior']
    del params_ph2['index']
    del params_ph2['log_likelihood']
    del params_ph2['log_prior']

    # extract the maximum likleihood value
    curve_lines_max = model(x_est,**params_ph2)
    
    return curve_lines_max, np.array(psd_keeper),rand_samples_2nd,log_prior 


def get_x_est(x):
    '''
    This function is making sure the frequency_resolution required by the user fits to the duration
    if not, we need to estimate our noise model in a different steps, which are then calculated   
    '''
    x_est=x.copy()
    if np.abs(config['frequency_resolution'] - 1/config['duration'])>0.01:
        x_est=np.linspace(x[0],x[-1],1+int(np.round((x[-1]-x[0])/config['frequency_resolution']))) 
    return x_est


# start of the script 

detectors = ["H1", "L1","V1"]

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 2:
    print("Usage: didnt find a config file:  python my_script.py ASD_config.json")
    sys.exit(1)

# Get the path to the configuration file from the command-line arguments
config_file_path = sys.argv[1]
with open(config_file_path, 'r') as file:
    config = json.load(file)


# handle configuration file 
config =asd_utilies.handle_config(config)

# settings, probably a more comprensive validation is needed: 
det=config['det']
if det not in detectors:
    print('mmm, not sure how break this to you, but you provided me with a funny detector name. See you later mate!')
    sys.exit()
    
user_label =config['user_label']
trigger_time = config['trigger_time']
maximum_frequency = config['maximum_frequency']
minimum_frequency = config['minimum_frequency']
roll_off = config['roll_off']  # Roll off duration of tukey window in seconds, default is 0.4s
duration = config['duration']  # Analysis segment duration
post_trigger_duration = config['post_trigger_duration']  # Time between trigger time and end of segment
resume = config['resume']
# these are a by product of the settings 
label = 'tPowerBilby_'+user_label+'_'+det 
outdir='Runs/'+label+'/'+config['outdir']
# create the directory 
asd_utilies.check_folder_and_open(outdir)
asd_utilies.check_folder_and_open(outdir+'/Validations')
asd_utilies.check_folder_and_open(outdir+'/Output')
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration
psd_duration = duration


# first thing would be to run the preprocesssing 
PreProcesstPowerBilby_class = PreProcessingtPowerBilby.PreProcesstPowerBilby(det=det,
                                                                             gw_event_start_time=config['pre_processing_end_time'], 
                                                                             duration=config['duration'],
                                                                             f_i=config['minimum_frequency'],
                                                                             f_f=config['maximum_frequency'],
                                                                             lines_low_freq_limit=config['low_freq_limit_detector'],
                                                                             spliting_min_freq_spacing=config['min_freq_spacing'],
                                                                             splitting_min_freq_segment=config['min_freq_segment'],
                                                                             n_steps_back=config['pre_processing_n_looking_back'],
                                                                             max_lines=config['n_lines'],
                                                                             label=label,
                                                                             resume= resume,
                                                                             debug=config['debug'],
                                                                             fake_data=False)



no_lines_curve=0
N_samples =config['N_noise_samples']
nlive=config['N_live_points']

x,y= asd_data_manipulations.get_GW_data(config,start_time)


xx_main,yy_main = PreProcesstPowerBilby_class.GetLinesXY_prior()
freq_split_vec = PreProcesstPowerBilby_class.GetFrequency_SplitPoints()
n_lines_per_split = PreProcesstPowerBilby_class.GetNlines_inEach_SplitPoints()
I_keep_data_indices = PreProcesstPowerBilby_class.GetHQ_dataPoints_Incides()

# save some data out
np.save(outdir+'/Output/'+label+'_keep_indices.npy', I_keep_data_indices)
print(('keep indicaes shape ',I_keep_data_indices.shape ))

if config['fit_entire_data']==True:
     I_keep_data_indices = [True] * len(I_keep_data_indices)
     I_keep_data_indices = np.array(I_keep_data_indices)

y_peaks_int,fit_lines_prior = FPH.process_lines_prior(gap_threshold=10,x=x,x_peaks=PreProcesstPowerBilby_class.lines_prior_x
                                                              ,y_peaks=PreProcesstPowerBilby_class.lines_prior_y )


I_FPH = y_peaks_int == 0 
np.save(outdir+'/Output/'+label+'_smooth_indices.npy', I_FPH)
print(('keep smooth indicaes shape ',I_FPH.shape ))


x_est= get_x_est(x)

sampled_1st,no_lines_curve,model_smooth,df_samples_1st = FPH.run_PL_fit(x=x[I_FPH],y=y[I_FPH],x_est=x_est,welch_y=PreProcesstPowerBilby_class.welch_y[I_FPH],
                                                                        preprocess_cls=PreProcesstPowerBilby_class,outdir=outdir,label=label,num_of_samples=N_samples,
                                                                        resume=resume,n_exp=config['n_exp'])
skip_samples_writing=config['skip_samples_writing']
# save hybrid samples 
hybrid_samples=[]
if len(x_est)==len(x):
    max_like =no_lines_curve.copy()
    max_like[~I_FPH] =PreProcesstPowerBilby_class.welch_y[~I_FPH]

    for sample in sampled_1st:
        tmp_sample = sample.copy()
        tmp_sample[~I_FPH] =PreProcesstPowerBilby_class.welch_y[~I_FPH]
        hybrid_samples.append(tmp_sample.copy())

    filename=outdir+'/Output/First_stage_hybrid_asd_samples_'+label+'.pkl'
    filename_max=outdir+'/Output/First_stage_hybrid_max_like_asd_sample_'+label+'.pkl'

    if not skip_samples_writing:
        with open(filename, 'wb') as f:
                    pickle.dump(hybrid_samples, f)
        with open(filename_max, 'wb') as f:
                    pickle.dump(max_like, f)            

        print('done writing 1st stage results ')                   


else:
     print('WARNING:: Couldnt write hybrid samples since the requested frequency_resolution != 1/Duration')        


ddddd

# this is done in the local fit level 
input_sections = list(np.arange(len(freq_split_vec)-1))# the number of section is one less than the number of points 
bool_flags = [resume] * len(input_sections)
#no_lines_curve = no_lines_curve[I_keep_data_indices]
with concurrent.futures.ProcessPoolExecutor() as executor:
        # Run the fit in parallel with different labels
        results = list(executor.map(run_fit_on_segment, input_sections,bool_flags))
print('Done sampling ')
final_max_curve = no_lines_curve.copy()    
final_sampled =  np.array(sampled_1st.copy())   

max_curve_NB = np.zeros(final_max_curve.shape)
final_sampled_NB = np.zeros((len(final_sampled),len(final_sampled[0])))

collect_dfs=[]

for result in results:
        curve_max, sampled_2nd,df_2nd,log_prior  = result  # Unpack the returned tuple
                               
        final_sampled_NB[:,~I_FPH] = np.maximum(final_sampled_NB[:,~I_FPH] ,sampled_2nd[:,~I_FPH])
             
        final_max_curve[~I_FPH]  = np.max([final_max_curve[~I_FPH] ,curve_max[~I_FPH]],axis=0)  # take the maximum only in the relevant parts  
       
        collect_dfs.append(df_2nd.copy())

final_sampled[:,~I_FPH] = final_sampled_NB[:,~I_FPH]
print('done opening results ')
filename=outdir+'/Output/asd_samples_'+label+'.pkl'

if not skip_samples_writing:
    with open(filename, 'wb') as f:
                pickle.dump(final_sampled, f)

    print('done writing results ')                    


# now do the final analysis using the maximum likelihood 

PostProcessingtPowerBilby_class= PostProcessingtPowerBilby.PostProcessingtPowerBilby(
                                                           curve_1st_phase=no_lines_curve,
                                                           curve_two_phases=final_max_curve,
                                                           I_1st=I_FPH,
                                                           I_2nd=I_keep_data_indices,
                                                           det=config['det'],
                                                           gw_event_start_time=start_time, 
                                                           duration=config['duration'],
                                                           f_i=config['minimum_frequency'],
                                                           f_f=config['maximum_frequency'],
                                                           label=label,
                                                           outdir=outdir,
                                                           full_pkl_file=filename,
                                                           config=config,
                                                           sections_results=collect_dfs,
                                                           first_phase_res=df_samples_1st)




