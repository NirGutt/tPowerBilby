
import arviz as az
import inspect
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
import gwpy
from itertools import compress
import bilby 
import os
import numpy as np
import matplotlib.pyplot as plt
from bilby.core.prior import LogUniform
from . import asd_utilies 
from context import tbilby
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from .asd_utilies import logger

class TransdimensionalConditionalUniform_lamda_pre(tbilby.core.prior.TransdimensionalConditionalUniform):   
    def transdimensional_condition_function(self,**required_variables):
        # setting the mimmum according the the last peak value
            minimum = self.minimum
            if(len(self.lamda)>0): # handle the first mu case
                minimum = self.lamda[-1] # set the minimum to be the location of the last peak 
                           
            return dict(minimum=minimum)

def exp_pre(x,A,lamda):  
  xfunc = x.copy() # start from zero offset 
  return A*np.power(xfunc,lamda)

class myGaussianLikelihood_pre(bilby.Likelihood):
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
           #print('Removing sigma')

      

    def log_likelihood(self):
        
        model_parameters = {k: self.parameters[k] for k in self.function_keys if k != 'sigma'}        
        est_asd = self.function(self.x, **model_parameters)      
        
        log_l =  np.sum(-self.normalization*(self.y/est_asd)**2 -np.log(est_asd**2))
        
        
        return log_l 

def remove_small_islands(arr, N):
    # Create a copy of the array to avoid modifying the original
    arr = np.copy(arr)
    
    # Find the indices where the value changes
    change_indices = np.where(np.diff(arr.astype(int)))[0]
    
    # Append the start and end of the array to handle edges
    change_indices = np.concatenate(([0], change_indices + 1, [len(arr)]))
    
    # Loop through the segments defined by change indices
    for start, end in zip(change_indices[:-1], change_indices[1:]):
        # Check if the segment consists of True values
        if arr[start] and (end - start) <= N:
            arr[start:end] = False  # Set this segment to False
    
    return arr

def run_PL_fit(x,y,x_est,label,resume):
    n_exp_pre=5
   
    componant_functions_dict={}
    componant_functions_dict[exp_pre]=(n_exp_pre,'A','lamda')
   
    model = tbilby.core.base.create_transdimensional_model('model',  componant_functions_dict,returns_polarization=False,SaveTofile=False)
    
    priors_t = bilby.core.prior.dict.ConditionalPriorDict()
    priors_t['n_exp_pre'] = tbilby.core.prior.DiscreteUniform(1,n_exp_pre,'n_exp_pre')
    priors_t  = tbilby.core.base.create_plain_priors(LogUniform,'A',n_exp_pre,prior_dict_to_add=priors_t,minimum=1e-30, maximum=1e-13)  
    priors_t = tbilby.core.base.create_transdimensional_priors(TransdimensionalConditionalUniform_lamda_pre,'lamda',nmax=n_exp_pre,nested_conditional_transdimensional_params=['lamda'],conditional_params=[],prior_dict_to_add=priors_t,minimum=-10,maximum=2)
    
        
    likelihood = myGaussianLikelihood_pre(x,y, model)
    outdir='tPowerBilby_outdir_tmp'
    file_path = outdir+'/'+'pre_processing_PL'+label+'_result.json'
    logger(('checking if results are in ', file_path, ' resume is' , resume),inspect.currentframe().f_code.co_name)
    
    if resume and os.path.exists(file_path):     
        
        logger('loadigng results',inspect.currentframe().f_code.co_name)
    
        result = bilby.read_in_result(filename=file_path) 
    else:    

        logger('running sampling',inspect.currentframe().f_code.co_name)


        result = bilby.run_sampler(
                    likelihood=likelihood,
                    priors=priors_t,                  
                    sample='rwalk',
                    nlive=150,                
                    outdir=outdir,
                    label='pre_processing_PL'+label,
                    clean=True,
                    resume=resume,               
                    npool=16,
                    )
    
   

    params = result.posterior.sort_values(by='log_likelihood',ascending=False).iloc[0].to_dict()
    del params['log_likelihood']
    del params['log_prior']
        
    return result.posterior,model(x_est,**params),model



class PreProcesstPowerBilby():
    def __init__(self, det,gw_event_start_time, duration ,f_i,f_f,
                 lines_low_freq_limit=30,
                 spliting_min_freq_spacing=30,
                 splitting_min_freq_segment=100,
                 n_steps_back=32,
                 max_lines=20,
                 resume=False,
                 debug=False,fake_data=False,label=''):
        # general
        self.gw_event_start_time=gw_event_start_time
        self.det = det
        self.duration = duration
        self.n_steps_back = n_steps_back
        self.f_i=f_i
        self.f_f=f_f
        self.max_lines=max_lines
        self.label=label
        self.resume=resume
        
        # line prior input
        self.low_freq_limit=lines_low_freq_limit
        
        # splitting line input 
        self.spliting_min_freq_spacing=spliting_min_freq_spacing
        self.splitting_min_freq_segment=splitting_min_freq_segment
        
        # auxsillary input 
        self.debug =debug
        self.fake_data = fake_data
        
        # self data 
        self.welch_x=None
        self.welch_y=None
        

       
       
        self._run_analysis()
        
    def _run_analysis(self):
        
        
        self.welch_x,self.welch_y =  self.get_GW_data_asd_welch(psd_end_time=self.gw_event_start_time,
                                                                det=self.det,
                                                                duration=self.duration,
                                                                f_i=self.f_i,
                                                                f_f=self.f_f)
        
        self.lines_prior_x,self.lines_prior_y = self.create_peaks_prior(self.welch_x,
                                                                        self.welch_y,
                                                                        med=True,
                                                                        n=self.n_steps_back,
                                                                        low_freq_limit=self.low_freq_limit)  
        
        self.create_power_law(self.welch_x,
                              self.welch_y,
                              self.lines_prior_x,
                              self.lines_prior_y)
        
        
       
        self.Shaplets_Amp_x,self.Shaplets_Amp_y = self.construct_shaplets_max_Amp(self.gw_event_start_time,
                                                                                  self.det,
                                                                                  self.duration, 
                                                                                  self.f_i,
                                                                                  self.f_f)
        
    
    
        self.Lines_Amp_x,self.Lines_Amp_y_max,self.Lines_Amp_y_min  =self.construct_line_max_Amp(self.gw_event_start_time,
                                                                       self.det,
                                                                       self.duration,
                                                                       self.f_i,
                                                                       self.f_f)
    
    
    
        self.freq_split_points = self.split_freqency_band(self.f_i,
                                                          self.f_f,
                                                          self.spliting_min_freq_spacing,
                                                          self.splitting_min_freq_segment)
    
    
        self.region_lines = self.restrict_n_lines(max_lines=self.max_lines)

        self.region_lines=self.find_too_far_points()

        if self.debug:
            #save the image for reference 
            fig= plt.figure(1)
            fig.set_size_inches(20, 12)  # Width, Height in inches

            # replot lines 
            plt.subplot(2,3,1)
          
            plt.loglog(self.welch_x,self.welch_y,'-k')
            x_tmp,y_tmp =self.get_GW_data_by_trigger(self.det,self.gw_event_start_time+self.duration,self.duration, self.f_i,self.f_f)
            plt.loglog(x_tmp,y_tmp,'-b',alpha=0.5)
           
            plt.plot(self.lines_prior_x[self.lines_prior_y>0],np.interp(self.lines_prior_x[self.lines_prior_y>0],self.welch_x,self.welch_y),'x')    



            fig.tight_layout()
            plt.savefig('Runs/'+str(self.label)+'/outdir/Validations/PreProcessingtPowerBilby_output_'+str(self.label)+'.png')

    
    def GetLinesXY_prior(self):
        return self.lines_prior_x,self.lines_prior_y  
    def GetShaplets_Amp_prior(self):
        return self.Shaplets_Amp_x,self.Shaplets_Amp_y 
    def GetLines_Amp_prior(self):
        return self.Lines_Amp_x,self.Lines_Amp_y_max,self.Lines_Amp_y_min 
    def GetFrequency_SplitPoints(self):
        return self.freq_split_points
    def GetNlines_inEach_SplitPoints(self):
        return self.region_lines
    def GetHQ_dataPoints_Incides(self):
        return self.I_keep

    def find_too_far_points(self):
        # always keep teh low freqency side below 40 Hz 
        I_huge = (5*self.curve < self.welch_y) & (self.welch_x > 40) 
        # remove isolated points, this means the peak is very sharp we still want it  
        I_huge= remove_small_islands(I_huge, N=2) # this will remove only two in a row 
        I_keep = ~I_huge
        
        
        if self.debug:
            plt.figure(1)
            plt.subplot(2,3,6)
            x,y =  self.get_GW_data_by_trigger(self.det,end_time=self.gw_event_start_time+self.duration,duration=self.duration, f_i=self.f_i,f_f=self.f_f)
            plt.loglog(x,y,label='some data')
            plt.loglog(self.welch_x,self.welch_y,label='welch')
            plt.loglog(self.welch_x,self.curve,label='fit')
            plt.loglog(self.welch_x,5*self.curve,label='too big')       
            plt.loglog(x[I_huge],y[I_huge],'or',label='some data after removal')
            plt.legend()
        self.I_keep = I_keep
        return self.update_n_lines_numbers(I_huge,self.freq_split_points,self.lines_prior_x,self.lines_prior_y  ,self.region_lines)
        
       
    
    def update_n_lines_numbers(self,exclude_indices, sections, prior_x,prior_y, lines_per_section):    
        
    # Initialize an updated dictionary to keep the adjusted number of lines for each section
        adjusted_lines = {}
        interp_function = interp1d(prior_x, prior_y, bounds_error=False, fill_value="extrapolate")
        # Loop through each section and apply the adjustment
        for i in range(len(sections) - 1):
            start_freq, end_freq = sections[i], sections[i + 1]
            section_key = i  # Dictionary key for the current section
            
            # Get indices corresponding to the frequency range in the current section
            section_indices = np.where((self.welch_x >= start_freq) & (self.welch_x < end_freq))[0]
            #print((start_freq, end_freq) )
            
            #print(len(section_indices))
            # Extract the prior values for this section
            section_prior = interp_function(self.welch_x[section_indices])
            
            # Extract the exclude indices (as a boolean mask) for this section
            section_exclude_indices = exclude_indices[section_indices]
            
            # Calculate the number of points in the section
            total_points_in_section = len(section_prior)
            
            # Calculate the number of included points and excluded points
            included_points = np.sum(section_prior)  # Points included according to the prior (prior_y == 1)
            excluded_points = np.sum(section_exclude_indices)  # Points excluded based on exclude_indices
            
            logger(('freq range ',start_freq, end_freq),inspect.currentframe().f_code.co_name)
            logger(('Total prior',included_points),inspect.currentframe().f_code.co_name)
            logger(('Total excluded_points',excluded_points),inspect.currentframe().f_code.co_name)
            # Calculate the percentage of points excluded
            percentage_removed = excluded_points / included_points
            
            # Adjust the number of lines in proportion to the percentage of points excluded
            current_lines = lines_per_section.get(section_key, 0)  # Get current number of lines for the section
            adjusted_lines_count = int(current_lines * (1 - percentage_removed))
            
            # Ensure there's at least 1 line if some points are still included
            adjusted_lines_count = max(adjusted_lines_count, 5) 
            
            # Update the adjusted lines dictionary
            adjusted_lines[section_key] = adjusted_lines_count
            
            
        if self.debug:
            regions = list(adjusted_lines.keys())
            line_counts = list(adjusted_lines.values())
            
            plt.subplot(2,3,5)
            plt.bar(regions, line_counts, color='r',alpha=0.7)
            
            # Add values on top of bars
            for i, count in enumerate(line_counts):
                plt.text(i, count + 1, str(count), ha='center')
                
            
        return adjusted_lines
        
    def create_power_law(self,x,y,x_peaks,y_peaks):
        # remove lines 
        
        gap_threshold = 10

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
        
        
        I = y_peaks_int == 0 
        x_d,y_d =self.get_GW_data_by_trigger(self.det,self.gw_event_start_time,self.duration, self.f_i,self.f_f)           
        
        plt.figure()
        plt.loglog(x_d,y_d,'o',alpha=0.6)
        plt.loglog(x_d[I],y_d[I],'x')
        I_freq = x <1400 
        # always include the low freq part 
        
        Ilow = x <50   
        #I[Ilow]=True
        fit_lines_prior = lines_prior.copy()
        fit_lines_prior[~Ilow]=0
        
            
        df,curve,model  = run_PL_fit(x=x_d[I&I_freq],y=y_d[I&I_freq],x_est=x_d,label=self.label,resume= self.resume)
        self.curve = curve 
    
    def split_freqency_band(self,minimum_frequency,maximum_frequency,min_freq_spacing=30,min_freq_segment=100):
    
    
        xx,yy = self.GetLinesXY_prior()
    
        
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
        #print('Initial splitting ASD points, verifying: ', natural_split_points)
        logger(('Initial splitting points, verifying: ', natural_split_points),inspect.currentframe().f_code.co_name)
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
        
        # verify that we have lines prior volume between each set of points
        remove_empty_points=[]
        
        for i in np.arange(len(natural_split_points)-1): 
            
            dV = np.interp(natural_split_points[i+1],xx,np.cumsum(yy))-np.interp(natural_split_points[i],xx,np.cumsum(yy))
            if dV>0:
                continue
            # if we are here that mean no volume space exssits, we dont need this part..      
            if i==0:
               
                logger('we got no lines in the first section, that a major probelm, lets kill it for now',
                       inspect.currentframe().f_code.co_name,'ERROR')
       
                
                fffff
           
            logger(('no line prior volume in that section, lets remove the barrior between the worlds! by removing point '),
                       inspect.currentframe().f_code.co_name)       
            remove_empty_points.append(i)
        natural_split_points = np.delete(natural_split_points, remove_empty_points)
        # aftere filtering 
        if self.debug:
            plt.subplot(2,3,4)
            x,y =self.get_GW_data_by_trigger(self.det,self.gw_event_start_time,self.duration, self.f_i,self.f_f)           
            plt.loglog(x,y)
            for p in natural_split_points:
                plt.vlines(p,0,10**(-18),'k')
            
        
        logger(('splitting the range into these segments: ', natural_split_points),inspect.currentframe().f_code.co_name)
       
        return natural_split_points
        
    def restrict_n_lines(self,max_lines):
        
        xx,yy = self.GetLinesXY_prior()
        freq_splits = self.GetFrequency_SplitPoints()
        prior_vol={}
        # calculate the precentage 
        Vtot = np.sum(yy[yy>0])
        for i in np.arange(len(freq_splits)-1):
            Ix = (xx > freq_splits[i]) & (xx < freq_splits[i+1])     
            yy_part = yy[Ix]            
            prior_vol[i]= np.sum(yy_part[yy_part>0])/Vtot
        
        max_prior = max(prior_vol.values())
        assigned_lines = {}
        
        for segment, prior_percentage in prior_vol.items():
            # Calculate the number of lines for the current segment based on the ratio
            lines = int((prior_percentage / max_prior) * max_lines)
            lines = int(np.ceil((lines + 2) / 5) * 5)
            assigned_lines[segment] = min(int(lines), max_lines)
      
        
        if self.debug:
            regions = list(assigned_lines.keys())
            line_counts = list(assigned_lines.values())
            
            plt.subplot(2,3,5)
            plt.bar(regions, line_counts, color='skyblue')
            plt.xlabel('Regions')
            plt.ylabel('Number of Lines')
            plt.title('Assigned Number of Lines to Regions Based on Prior Percentage')
            
            # Add values on top of bars
            for i, count in enumerate(line_counts):
                plt.text(i, count + 1, str(count), ha='center')
            
            
        
        
        return assigned_lines
        
    def create_peaks_prior(self,x,y,med=False,n=32,low_freq_limit=30): # this is the new version including the edges due to a problem seen :(
        
    
    
        Ilow =x < low_freq_limit   
        
        arr = np.zeros(x.shape)
        arr_new = np.zeros((2*x.shape[0],))
        dx = x[1]-x[0]
        x_new = np.arange(x[0],x[-1]+dx,0.5*dx)
        
        
        padded_data = np.pad(y, 50, mode='edge')
            
        y_med = asd_utilies.med_rolling(padded_data, 101)
        
        # low freq detector
        
        ransac = RANSACRegressor(estimator=PowerLawModel(), min_samples=2,random_state=42)
        ransac.fit(x[Ilow].reshape(-1, 1), y[Ilow].reshape(-1, 1))
       
        # Extract the estimated parameters
        A = ransac.estimator_.A_
        alpha = ransac.estimator_.alpha_
        if self.debug:
            logger(('Estimated spectral index ', alpha),inspect.currentframe().f_code.co_name)
       
        
        y_med_orig=y_med.copy()
        y_med[Ilow]= A*x[Ilow]**(alpha)
        sigmas = np.arange(6,2.0,-0.5)
        if med:
            sigmas/=np.sqrt(n)
        for sigma in sigmas:
            I = y > y_med + y_med*sigma    
            arr[I] += 1
         
           
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
            
        if self.debug:
            plt.figure(1)
            plt.subplot(2,3,1)
            plt.loglog(x,y,'-k')
            x_tmp,y_tmp =self.get_GW_data_by_trigger(self.det,self.gw_event_start_time+self.duration,self.duration, self.f_i,self.f_f)
            plt.loglog(x_tmp,y_tmp,'-b',alpha=0.5)
            plt.loglog(x,y_med_orig,'-.r')
            plt.loglog(x,y_med,'-m')
            plt.plot(x_new[arr_new>0],np.interp(x_new[arr_new>0],x,y),'x')
    
        # set the prior to equal values for all peaks
        arr_new[arr_new>0]=1
        return x_new,arr_new
    
    
    def get_GW_data_by_trigger(self,det,end_time,duration, f_i,f_f):
            
    
    
           
            start_time = end_time - duration
          

            psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(det, start_time, end_time)
            
            if self.fake_data and self.debug==True:
                sampling_frequency=4096
                ifos = bilby.gw.detector.InterferometerList([det])
                for ifo in ifos:
                    ifo.minimum_frequency = self.f_i
                    ifo.maximum_frequency = self.f_f
                ifos.set_strain_data_from_power_spectral_densities(
                        sampling_frequency=sampling_frequency,
                        duration=duration,
                        start_time=start_time)
                psd_data = gwpy.timeseries.TimeSeries(data= ifos[0].strain_data.time_domain_strain,t0=start_time,dt=1/sampling_frequency)
            
            
            
            ifo = bilby.gw.detector.get_empty_interferometer(det)    
            ifo.strain_data.set_from_gwpy_timeseries(psd_data)
            ifo.maximum_frequency = f_f
            ifo.minimum_frequency = f_i
            
            x_ifo = ifo.strain_data.frequency_array
            y_ifo = ifo.strain_data.frequency_domain_strain
            
            I = (x_ifo >= f_i) &  (x_ifo <= f_f) 
    
            
            return x_ifo[I],np.abs(y_ifo[I])
    
    
    def get_GW_data_asd_welch(self,psd_end_time,det='L1',duration=4,f_i=20,f_f=896):
      
        roll_off=0.4 
        psd_duration = self.n_steps_back * duration
        psd_start_time = psd_end_time - psd_duration
        psd_end_time = psd_end_time
        
        
            
        #,channel='GWOSC-4KHZ_R1_STRAIN'
        psd_data = gwpy.timeseries.TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
        
        if self.fake_data and self.debug==True:
            sampling_frequency=4096
            ifos = bilby.gw.detector.InterferometerList([det])
            for ifo in ifos:
                ifo.minimum_frequency = self.f_i
                ifo.maximum_frequency = self.f_f
            ifos.set_strain_data_from_power_spectral_densities(
                    sampling_frequency=sampling_frequency,
                    duration=psd_duration,
                    start_time=psd_start_time)
            psd_data = gwpy.timeseries.TimeSeries(data= ifos[0].strain_data.time_domain_strain,t0=psd_start_time,dt=1/sampling_frequency)
        
        
        psd_alpha = 2 * roll_off / duration
        psd = psd_data.psd(
            fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="welch"
        )
        #requency_array=psd.frequencies.value, psd_array=psd.value
        
        psd_frequencies = psd.frequencies.value
        psd = psd.value
        I = (psd_frequencies >= f_i) &  (psd_frequencies <= f_f) 
        
        
        return psd_frequencies[I],np.sqrt(psd[I])
    
    
    def construct_shaplets_max_Amp(self,gw_event_start_time,det,duration =4 , f_i=20.0,f_f=896.0):
        
        
        if self.welch_x is None or self.welch_y is None:
        
            self.welch_x,self.welch_y = self.get_GW_data_asd_welch(self.gw_event_start_time,self.det,self.duration,self.f_i,self.f_f)
            
        padded_data = np.pad(self.welch_y, 50, mode='edge')
        y_med = asd_utilies.med_rolling(padded_data, 101)
        dx = self.welch_x[1]-self.welch_x[0]
        factor =np.sqrt(4*dx) # this turn the data into ASD units - for 4s it is just 1
        Safety_factor =1.1 
        
        Amp_max= Safety_factor* 3.5*(y_med)/factor
        
        if self.debug:
            plt.figure(1)
            plt.subplot(2,3,2)
            x,y =  self.get_GW_data_by_trigger(det,end_time=gw_event_start_time+duration,duration=duration, f_i=f_i,f_f=f_f)
            plt.loglog(x,y,label='some data')
            plt.loglog(self.welch_x,self.welch_y,label='Welch')
            plt.loglog(self.welch_x,Amp_max,label='Shapletes Max Amp')
            plt.legend()
        
        
        return self.welch_x.copy(),Amp_max
       
        
  

    
    def construct_line_max_Amp(self,start_time,det,duration =4 , f_i=20.0,f_f=896.0):
        # take 32 time segments
        N_seg=self.n_steps_back
        data=[]
        for N in np.arange(N_seg):
            if self.debug:
                logger('getting data for max '+str(N)+'/'+str(N_seg),inspect.currentframe().f_code.co_name)
        
            x,y=self.get_GW_data_by_trigger(det,start_time-N*self.duration,self.duration , f_i,f_f)
            data.append(y)
    
        max_data=np.max(np.vstack(data),axis=0) + self.curve 
        min_data=np.min(np.vstack(data),axis=0) - self.curve 
        min_data[min_data<self.curve] = self.curve[min_data<self.curve] # if we go negative we set a minumu of 1*std 
        # this doesnt make any sense lets get ride of these points
        min_data[min_data>max_data]=0.9 # that no good 
        max_data[min_data>max_data]=1.0# set somethign slightly higher 
        # but we should get ride of these completely 
        
        if self.debug:
            plt.figure(1)
            plt.subplot(2,3,3)
            x,y =  self.get_GW_data_by_trigger(det,end_time=start_time+duration,duration=duration, f_i=f_i,f_f=f_f)
            plt.loglog(x,y,label='some data')
            plt.loglog(self.welch_x,self.welch_y,label='Welch')
            plt.loglog(x,max_data,label='Lines Max Amp',alpha=0.5)
            plt.loglog(x,min_data,label='Lines Min Amp',alpha=0.5)
            plt.legend()
        return x,max_data,min_data






class PowerLawModel(BaseEstimator, RegressorMixin):
    def __init__(self ,A_=None, alpha_=None):
        self.A_ = A_
        self.alpha_ = alpha_ 
    
    def fit(self, X, y):
        # Log-transform the data to linearize the power-law relationship
        log_X = np.log(X)
        log_y = np.log(y)
        
        # Fit a linear model to the transformed data
        linear_model = np.polyfit(log_X.ravel(), log_y, 1)
        
        # Extract parameters from the linear fit
        self.alpha_ = linear_model[0]
        self.A_ = np.exp(linear_model[1])
        return self
    
    def predict(self, X):
        return self.A_ * X**self.alpha_

    def get_params(self, deep=False):
        return {"A_": self.A_, "alpha_": self.alpha_}




