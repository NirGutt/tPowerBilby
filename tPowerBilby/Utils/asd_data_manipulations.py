import numpy as np
from . import asd_utilies
import matplotlib.pyplot as plt
import gwpy
import bilby
import pickle 
import os.path
from scipy.signal import find_peaks


def create_peaks_prior(x,y,use_low_freq_detector=False,low_freq_limit=40): # this is the new version including the edges due to a problem seen :(
    

    Ilow =x < low_freq_limit   
    peaks, properties = find_peaks(y[Ilow], height=1e-23, distance=3)


    arr = np.zeros(x.shape)
    arr_new = np.zeros((2*x.shape[0],))
    dx = x[1]-x[0]
    x_new = np.arange(x[0],x[-1]+dx,0.5*dx)
    
    
    padded_data = np.pad(y, 50, mode='edge')
        
    y_med = asd_utilies.med_rolling(padded_data, 101)
    sigmas = np.arange(6,2.0,-0.5)
    for sigma in sigmas:
        I = y > y_med*sigma    
        arr[I] += 1
    
    if use_low_freq_detector==True and len(peaks)>0: # add the low range peaks if needed 
        print('using low freq peaks detector')
        arr[peaks]+=4


    x_old_inx = arr > 0
    x_vals = x[x_old_inx]
    x_old_inx = np.where(x_old_inx)[0]
    for x_v,x_i in zip(x_vals,x_old_inx):
        difference_array = np.absolute(x_new-x_v)
 
# find the index of minimum element from the array
        index = difference_array.argmin()
        arr_new[index] = arr[x_i]
        arr_new[index+1] = arr[x_i]
        arr_new[index-1] = arr[x_i]
    if use_low_freq_detector==True and len(peaks)>0:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(x_new,arr_new)
        
            plt.subplot(2,1,2)

            y_est = np.interp(x_new[arr_new>0], x, y)
            plt.loglog(x,y)
            plt.loglog(x_new[arr_new>0],y_est,'xk')
            plt.savefig('peaks_prior_with_low_freq_detector.png')


    return x_new,arr_new

def estimate_natural_parts(x,y,minimum_frequency,maximum_frequency,min_freq_spacing=30,min_freq_segment=100):
    from itertools import compress
    
    xx,yy = create_peaks_prior(x,y)

    
    I = yy>0
    full_range  =np.zeros(yy.shape)
    full_range[I]=1
 
    x_spaces = xx[I][1:]
    spaces = np.diff(xx[I])
    Ispaces =  spaces > min_freq_spacing # spaces larger than 30 Hz
    
    list_a = np.arange(len(Ispaces))
    inx = np.array(list(compress(list_a, Ispaces)))

    natural_split_points = 0.5*(x_spaces[inx]+x_spaces[inx-1])
    # add the boundries 
    natural_split_points =  np.append(natural_split_points,[minimum_frequency,maximum_frequency])
    inx =  np.append(inx,[0,int(np.round(xx[-1]))])


    remove_points = []
    natural_split_points = np.sort(natural_split_points)
    print(natural_split_points)
    for point in np.arange(1,len(natural_split_points)): # without the edges 
        prev= -1
        j=1
        # find teh last one we didnt remove 
        while prev < 0:
            if point-j==0:
                prev = point -j 
                break
            
            if point-j in remove_points:
                j+=1
                continue 
                
            else:
                prev = point -j 
            
        
        if natural_split_points[point]-natural_split_points[prev] < min_freq_segment:
            #if point == len(yy): 
            if point == len(natural_split_points)-1: # is it the last point ?     
                remove_points.append(prev)            
            elif point == 1:    
                remove_points.append(1)        
            elif spaces[point] > spaces[prev]:
                remove_points.append(prev)
            else:
                remove_points.append(point)
                
    natural_split_points = np.delete(natural_split_points, remove_points)
    
    # aftere filtering 
    plt.figure(100)
    plt.loglog(x,y)
    for p in natural_split_points:
        plt.vlines(p,0,10**(-18),'k')
        
    print('splitting the ASD range into these segments')
    print(natural_split_points)
    return natural_split_points

def determine_shaplet_effective_dis(shaplet_num,fit_params_in,model,x,plotit=False):
    
    
    if fit_params_in['n_shaplets'+str(shaplet_num)]==0: # there are no shapletes here!
        print('determine_shaplet_effective_dis:Got zero n_shaplets'+str(shaplet_num) + ' returning with zero ') 
        return 0 
    
    fit_params_no_shaplets= fit_params_in.copy()
    fit_params_one_shaplet= fit_params_in.copy() 
    fit_psd_params  = fit_params_in.copy() 
    
    # check which version we are dealign with 
    if 'n_lorentzian' in fit_psd_params:
        fit_psd_params['n_lorentzian']=0

    if 'n_improved_lorentzian' in fit_psd_params:
        fit_psd_params['n_improved_lorentzian']=0    
    
    
    for k in np.arange(4):
        fit_params_no_shaplets['n_shaplets'+str(k)]=0
        fit_psd_params['n_shaplets'+str(k)]=0
        if k==shaplet_num:
            continue
        fit_params_one_shaplet['n_shaplets'+str(k)]=0
           
    d_shaplet = model(x,**fit_params_one_shaplet) -  model(x,**fit_params_no_shaplets)
    asd = model(x,**fit_psd_params)
    if plotit:
        plt.figure() 
        plt.plot(x,d_shaplet)
    final_effcetive_sz = 5
    for effcetive_sz in np.arange(2,5,0.1):
        d_shaplet = model(x,**fit_params_one_shaplet) -  model(x,**fit_params_no_shaplets)
        I = (
            (x > fit_params_one_shaplet['s'+str(shaplet_num) + 'x_center'] -  effcetive_sz* fit_params_one_shaplet['s'+ str(shaplet_num)+'beta']) & 
            (x <  fit_params_one_shaplet['s'+str(shaplet_num) + 'x_center']+  effcetive_sz* fit_params_one_shaplet['s'+ str(shaplet_num)+'beta'])
            )
        d_shaplet[I] =0 
        max_A = np.max(abs(d_shaplet)/asd)
        if max_A < 0.1:
            if plotit:
                plt.plot(x,d_shaplet)
            final_effcetive_sz = effcetive_sz
            print(effcetive_sz)
            break
    if plotit:    
        plt.plot(x,asd)
        
    
    return final_effcetive_sz


def find_ranges_to_remove(max_likelihood_params,model,x):
    # run over the lines and set sections to remove, f_i,f_f
    # run over teh shaplets and set sections to remove, f_i,f_f
    ranges_to_remove=[]
    
    # check which lorentzian flavour we used: 
    total_num_lines= 0    
    if 'n_improved_lorentzian' in max_likelihood_params.keys(): 
        total_num_lines = max_likelihood_params['n_improved_lorentzian']
    else:                                            
        total_num_lines = max_likelihood_params['n_lorentzian']
    
    for line_num in np.arange(total_num_lines):
        l_n= str(int(line_num))
                
        s_f =max_likelihood_params['loc'+l_n]-(max_likelihood_params['loc'+l_n]/50)*1.1 # this might change to a more robust estimation based on teh new fitted parameters 
        e_f =max_likelihood_params['loc'+l_n]+(max_likelihood_params['loc'+l_n]/50)*1.1
        ranges_to_remove.append((s_f,e_f))
    
    for n_shap in np.arange(4):
        if max_likelihood_params['n_shaplets'+str(n_shap)]>0:

            eff_sz = determine_shaplet_effective_dis(n_shap,max_likelihood_params,model,x,plotit=False)

            # eff_sz used to be a fixed value of 1.5, which was wrong 
            s_f = max_likelihood_params['s'+str(n_shap)+'x_center']-\
                eff_sz* max_likelihood_params['s'+str(n_shap)+'beta'] 
            e_f = max_likelihood_params['s'+str(n_shap)+'x_center']+\
                eff_sz* max_likelihood_params['s'+str(n_shap)+'beta']     
            ranges_to_remove.append((s_f,e_f))
        
    return ranges_to_remove   

def clean_data_from_dirty_dirty_points(x,y,list_max_likelihood_params,model_in):
    
    full_range_list=[]
    
    for max_likeli in list_max_likelihood_params:
        tmp_list = find_ranges_to_remove(max_likeli,model_in,x) 
  
        full_range_list.extend(tmp_list)
   
    Inx_to_remove=[]
    for pair in full_range_list:
       
       I = np.where((x> pair[0]) & (x < pair[1]))[0].tolist()
       if (len(I)>0):
           Inx_to_remove.extend(I)
       
    #remove same inx occuring twice     
    Inx_to_remove = np.unique(Inx_to_remove)
          
    if len(Inx_to_remove)>0:   
        x_filtered = np.delete(x, Inx_to_remove)    
        y_filtered = np.delete(y, Inx_to_remove)    
        return x_filtered,y_filtered  
    else:
        return x,y

def get_GW_data_asd_welch(psd_end_time,det='L1',duration=4,f_i=20,f_f=896,method='welch',n_step_back=32,roll_off=0.4):
                                
        psd_duration = n_step_back * duration
        psd_start_time = psd_end_time - psd_duration
        psd_end_time = psd_end_time
            
        psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
                                    
        psd_alpha = 2 * roll_off / duration
        psd = psd_data.psd(
            fftlength=duration, window=("tukey", psd_alpha), method=method
        )
                        
        psd_frequencies = psd.frequencies.value
        psd = psd.value
        I = (psd_frequencies >= f_i) &  (psd_frequencies <= f_f) 
            
            
        return psd_frequencies[I],np.sqrt(psd[I]) 


def get_GW_data(config,start_time,f_i=None,f_f=None,return_raw_data=False):
    
    minimum_frequency = config['minimum_frequency']
    maximum_frequency= config['maximum_frequency']
    psd_duration= config['duration']
    det = config['det']
    roll_off=config['roll_off']


    if f_i is None:
        f_i =minimum_frequency
    if f_f is None:
        f_f=maximum_frequency  

    # making sure we are at a reasnable range 
    if minimum_frequency > f_i:
        f_i =minimum_frequency 
    if maximum_frequency < f_f:
        f_f = maximum_frequency 

    
    psd_start_time = start_time - psd_duration
    psd_end_time = start_time

    key = det+'_'+str(psd_start_time)+'_'+str(psd_end_time)+'_'+str(f_i)+'_'+str(f_f)
   
  
    psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)    

    ifo = bilby.gw.detector.get_empty_interferometer(det)    
    
    ifo.strain_data.roll_off = roll_off
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    

    ifo.strain_data.set_from_gwpy_timeseries(psd_data)
    
    x_ifo = ifo.strain_data.frequency_array
    y_ifo = ifo.strain_data.frequency_domain_strain
    
    Ew = np.sqrt(ifo.strain_data.window_factor)

    I = (x_ifo >= f_i) &  (x_ifo <= f_f) 

    if return_raw_data:
            return x_ifo[I],y_ifo[I]/Ew
    
    return x_ifo[I],np.abs(y_ifo[I])/Ew

    