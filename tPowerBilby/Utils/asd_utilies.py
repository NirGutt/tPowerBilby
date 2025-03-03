import numpy as np
import matplotlib.pyplot as plt
import os 
import inspect
def logger(message, function_str, level='Info'):
    # simple logger like 
    # Ensure inputs are strings
    message = str(message)
    function_str = str(function_str)
    level = str(level)  

    
    print(f"tPowerBilby::[{level}]::{function_str}: {message}")




def check_folder_and_open(folder_path):

    logger('checking output folder ',inspect.currentframe().f_code.co_name)
    logger(folder_path,inspect.currentframe().f_code.co_name)

    if not os.path.exists(folder_path):
    # If it doesn't exist, create it
        os.makedirs(folder_path)
        logger('we didnt find it , so we created one ',inspect.currentframe().f_code.co_name)


def max_rolling(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.max(rolling,axis=axis)
    
def min_rolling(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.min(rolling,axis=axis)

def med_rolling(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.median(rolling,axis=axis)    

def check_data(x,y,x_clean,y_clean,freq_split_vec,section,round_num,outdir):
        plt.figure()
        plt.loglog(x,y,'k')
        plt.loglog(x_clean,y_clean,'r')
        plt.vlines(freq_split_vec[section-1],10**(-25),10**(-17))
        if section==len(freq_split_vec):
            plt.vlines(freq_split_vec[-1],10**(-25),10**(-17))
        else :   
            plt.vlines(freq_split_vec[section],10**(-25),10**(-17))
        plt.savefig(outdir+'/clean_data_section_'+str(section)+ '_round_num_' +str(round_num)+'.png')

     

def check_from_where_to_start(freq_split_vec,number_of_rounds,label,outdir):
    import os 
    for r in np.arange(number_of_rounds):    
        for section in np.arange(1,len(freq_split_vec)+1):
            
            if section==len(freq_split_vec):
                label_sampling_result=outdir+'/'+label+'_round_'+str(r)+'_'+ str(freq_split_vec[0])+'_'+str(freq_split_vec[-1])+'_full_range_result.json'
            else:
                label_sampling_result=outdir+'/'+label+'_round_'+str(r)+'_'+ str(freq_split_vec[section-1])+'_'+str(freq_split_vec[section])+'_result.json'
            if not os.path.exists(label_sampling_result):
                return section,r 
        
        
    return -1,-1 

def handle_config(config):
    
    # defualt settings 
    standarized_config={}
    # GW settings 
    standarized_config['det']='H1' 
    standarized_config['split_run'] = True    
    standarized_config['trigger_time'] = 1126259462.4    
    standarized_config['maximum_frequency'] = 896
    standarized_config['minimum_frequency'] = 20    
    standarized_config['roll_off'] = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
    standarized_config['duration'] = 4  # Analysis segment duration
    standarized_config['frequency_resolution'] = 1/standarized_config['duration']
    standarized_config['post_trigger_duration'] = 2  # Time between trigger time and end of segment
    
    # pre processing settings 
    standarized_config['pre_processing_n_looking_back']=32
    standarized_config['min_freq_spacing'] =30
    standarized_config['min_freq_segment'] =100
    standarized_config['low_freq_limit_detector'] =40
    # all the priors construction is done before that GPS time 
    standarized_config['pre_processing_end_time'] =standarized_config['trigger_time'] + standarized_config['post_trigger_duration'] - standarized_config['duration'] 
    
    # sampling settings 
    standarized_config['n_exp'] =5 
    standarized_config['n_lines'] =20
    standarized_config['user_label'] ='GW150419'
    standarized_config['outdir']='outdir'
    standarized_config['resume']=True
    standarized_config['fit_entire_data']=True
    standarized_config['N_noise_samples']=1000
    standarized_config['skip_samples_writing']=False
    standarized_config['N_live_points']= 700

    # prior settings 
    # oen can't set teh first one since it causes issue with teh sampling, this is a limitation for now  
    standarized_config['alpha1']=-100.0
    standarized_config['alpha2']=-100.0
    standarized_config['alpha3']=-100.0
    standarized_config['alpha4']=-100.0

    standarized_config['lines_prior']=list([])


    standarized_config['debug']= True


    for k in standarized_config.keys():
        if k in config.keys(): # the user doesn like teh default settings :(
            desired_type = type(standarized_config[k])
            if k=='split_run' or k=='use_simpler_lorenztain_in_round_0' or k=='use_low_freq_detector' or k =='use_gaussian_prior_for_lorenztain_tail' or k=='resume' or k=='fit_entire_data' or k=='skip_samples_writing' or k=='debug':
                standarized_config[k] = config[k]# this is a bool, treat it acordingly 
            else:     
                standarized_config[k]=desired_type(config[k]) # make sure th user provided with th e right type 
    
    
    logger('The ASD tBilby run configuration is as follows, make sure you are happy with it:',inspect.currentframe().f_code.co_name)    
    logger(standarized_config,inspect.currentframe().f_code.co_name)
    return standarized_config


