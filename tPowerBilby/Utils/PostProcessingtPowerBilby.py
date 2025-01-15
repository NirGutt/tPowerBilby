import bilby 
import numpy as np
import gwpy
from . import asd_data_manipulations
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import h5py
import inspect
from scipy.stats import anderson
from .asd_utilies import logger
mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['DejaVu Serif']
mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['text.usetex'] = True


class PostProcessingtPowerBilby():    
    def __init__(self, curve_1st_phase, curve_two_phases,I_1st,I_2nd,
                 det,gw_event_start_time, duration ,f_i,f_f,label,outdir,full_pkl_file,config,sections_results,first_phase_res,
                 PE_file=None):
        
        self.gw_event_start_time=gw_event_start_time
        self.det = det
        self.duration = duration        
        self.f_i=f_i
        self.f_f=f_f
        self.label=label
        self.curve_1st_phase=curve_1st_phase # this is the broadband curve
        self.curve_two_phases=curve_two_phases # this is the broadband+narrowband curve
        self.outdir=outdir
        self.pkl_file=full_pkl_file # final samples file
        self.config= config
        self.I_1st=I_1st # indices used to fit the braodband curve
        self.I_2nd=I_2nd # indices used to fit the braodband+narrowband curves
        self.post_ouput_str='PostProcessingtBilby_'        
        self.sections_results=sections_results # this is the narrowband results
        self.first_phase_res=first_phase_res # this is the broadband result
        self.low_freq_limit=40
        self.PE_file= PE_file # LIGO parameter estimation file
        
 
        self.save_max_like()
        self.Create_HQ_mask()

        self.validate_output()        
        
        self.simple_plot()                
        self.plot_Normal_dist()
        self.plot_n_lines()
        self.count_discrete()

    def validate_output(self):

        x,psd_data= asd_data_manipulations.get_GW_data(self.config,self.gw_event_start_time)
        x_welch,y_welch = asd_data_manipulations.get_GW_data_asd_welch(psd_end_time=self.gw_event_start_time,det=self.config['det'],
                                                    duration=self.config['duration'],                                                    
                                                    f_i=self.config['minimum_frequency'],
                                                    f_f=self.config['maximum_frequency'],
                                                    method ='welch')
         
     
        def plot_file(file_name,color,label):
            with open(file_name, 'rb') as file:
                data = pickle.load(file)
                
            if data.shape[1]==1:
                data=data.T 
            plt.loglog(x,np.median(data,axis=0),'-'+color,label=label,alpha=0.4)   

        def plot_inx(file_name,color,label,x,y):
            # plotting the NOT indices
            Inxs = np.load(file_name)
            plt.loglog(x[~Inxs],psd_data[~Inxs],'o'+color,alpha=0.4,label=label)
        
        sz_F=2
        plt.figure(figsize=(sz_F*6.18, sz_F*3.8)) #  
        plt.loglog(x,psd_data,'k',label='data')
        # load indices and plot them 
        
        filename=self.outdir+f'/Output/{self.label}_keep_indices_post_processing.npy'   
        plot_inx(filename,'r','Low Quality Indices Post Processing',x,psd_data)
        
        filename=self.outdir+f'/Output/{self.label}_keep_indices.npy'   
        plot_inx(filename,'m','Low Quality Indices Pre Processing',x,psd_data)

        filename=self.outdir+f'/Output/{self.label}_smooth_indices.npy'   
        plot_inx(filename,'b','Lines Indices',x,psd_data)

        plt.legend()
        plt.savefig(self.outdir+f'/Validations/asd_samples_Indices.png',dpi= 300,bbox_inches="tight" )



        plt.figure(figsize=(sz_F*6.18, sz_F*3.8)) #  
        plt.loglog(x,psd_data,'k',label='data')

        
        
        plt.loglog(x_welch,y_welch,'-r',alpha=0.6,label='welch')

          
        samples_fname=self.outdir+f'/Output/asd_samples_{self.label}.pkl'         
        plot_file(samples_fname,'c','median tPowerBilby')

        samples_fname=self.outdir+f'/Output/First_stage_hybrid_asd_samples_{self.label}.pkl'  
        plot_file(samples_fname,'c','Broadband hybrid median tPowerBilby')

        samples_fname=self.outdir+f'/Output/First_stage_hybrid_max_like_asd_sample_{self.label}.pkl'  
        plot_file(samples_fname,'c','Broadband hybrid max-like tPowerBilby')

        samples_fname=output_fname=self.outdir+f'/Output/{self.post_ouput_str}_max_like.pkl'  
        plot_file(samples_fname,'m','max-like tPowerBilby')


        plt.legend()
        plt.savefig(self.outdir+f'/Validations/asd_samples_median.png',dpi= 300,bbox_inches="tight" )


    def load_GWTC_curve(self):
            # locate the file automatically accordign to the name 
        if self.PE_file is None:
            return  None,None
        if not os.path.exists(self.PE_file):
            logger('couldnt locate file, settign it to None',inspect.currentframe().f_code.co_name)            
            return  None,None
            

        if self.PE_file is not None:            
            f = h5py.File(self.PE_file, 'r')
            PE_psds_dict = self._get_psd_from_ligo(f)

        return PE_psds_dict[self.det][0],np.sqrt(PE_psds_dict[self.det][1])


    def save_max_like(self):
        
        output_fname=self.outdir+f'/Output/{self.post_ouput_str}_max_like.pkl'

        with open(output_fname, 'wb') as f:
                    pickle.dump(np.array(self.curve_two_phases).reshape(-1,1), f)  

      

    def Create_HQ_mask(self):
        x,y= asd_data_manipulations.get_GW_data(self.config,self.gw_event_start_time)
        self.Ikeep = (y<=5*self.curve_1st_phase)  | (x<self.low_freq_limit)      
        
        logger(('removing low quality data: ', np.sum(self.Ikeep)-len(self.Ikeep) ),inspect.currentframe().f_code.co_name)

        if np.sum(self.I_2nd)<len(self.I_2nd):# meaning some thign was removed, lets combaine teh two
            self.Ikeep = self.Ikeep&self.I_2nd
            logger(('removing low quality data: ', np.sum(self.Ikeep)-len(self.Ikeep) ),inspect.currentframe().f_code.co_name)            
        np.save(self.outdir+'/Output/'+self.label+'_keep_indices_post_processing.npy',self.Ikeep)
   
    def count_discrete(self):
        
        from collections import Counter
        mean_n_lines=0 
        mean_n_sh=0
        book_keeper={}

        sh_arr = ( (self.first_phase_res['n_shaplets0']>0).values.astype(int)+
                   (self.first_phase_res['n_shaplets1']>0).values.astype(int)+
                   (self.first_phase_res['n_shaplets2']>0).values.astype(int)+
                   (self.first_phase_res['n_shaplets3']>0).values.astype(int)
                   )
        counter_sh = Counter(sh_arr)
        mean_n_sh, frequency = counter_sh.most_common(1)[0]        
        book_keeper[-1]=mean_n_sh

        for res,i in zip(self.sections_results,np.arange(len(self.sections_results))):            
            counter = Counter(res['n_improved_lorentzian'].values)
            most_frequent_value, frequency = counter.most_common(1)[0]
            mean_n_lines+=most_frequent_value
            
            sh_arr = ( (res['n_shaplets0']>0).values.astype(int)+
                   (res['n_shaplets1']>0).values.astype(int)+
                   (res['n_shaplets2']>0).values.astype(int)+
                   (res['n_shaplets3']>0).values.astype(int)
                   )
            counter_sh = Counter(sh_arr)
            n_sh, frequency = counter_sh.most_common(1)[0]             
            mean_n_sh+=n_sh
            book_keeper[i]=(most_frequent_value,n_sh)
      
        

        
        counter = Counter(self.first_phase_res['n_exp'].values)
        most_frequent_value, frequency = counter.most_common(1)[0]
        logger(('n_lines=', mean_n_lines, 'n_shap' , mean_n_sh ,'n_exp= ' ,most_frequent_value ),inspect.currentframe().f_code.co_name)       
        plt.figure()
        plt.hist(self.first_phase_res['n_exp'].values)
        plt.savefig(self.outdir+f'/Validations/{self.post_ouput_str}n_exp_dist{self.label}.png',bbox_inches="tight")


    def _get_psd_from_ligo(self,f):    
        # = f['C01:IMRPhenomXPHM']['psds']
        ret_dict={}
        psd_data= f['C01:IMRPhenomXPHM']['psds']
        for ifo_name in list(psd_data):
        
            psd_freq_array = psd_data[ifo_name][:,0]
            I = (psd_freq_array >= self.f_i) &  (psd_freq_array <= self.f_f)
            frq = psd_data[ifo_name][I,0]
            vals = f['C01:IMRPhenomXPHM']['psds'][self.det][I,1]
            ret_dict[ifo_name] = np.vstack(list([frq,vals]))
    
       
        return ret_dict 

    def plot_n_lines(self):
        lines_dist_arr=[]
        for res in self.sections_results: # these are dataframes
            lines_dist_arr.append(res['n_improved_lorentzian'])

        min_length = min(len(lst) for lst in lines_dist_arr)
        trimmed_lists = [lst[:min_length] for lst in lines_dist_arr]
        resulting_list = np.sum(trimmed_lists, axis=0)
        low=int(np.min(resulting_list))
        high=int(np.max(resulting_list))
        F=1
        F= int(F*9)
        sz_F=0.5
        plt.figure(figsize=(sz_F*6.18, sz_F*3.8)) #        
        bins=np.arange(low-5,high+5)
        args_hist={'hatch':'//','color':'k','alpha' : 0.4,'edgecolor':'none'}
        plt.hist(resulting_list, bins=bins,**args_hist)        
        plt.xlabel('$\mathrm{N}_{\mathrm{line}}$',fontsize=F,fontfamily='Times New Roman')
       
        
        ax= plt.gca()
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.tick_params(axis='both', labelsize=F) 
        plt.grid(True,linestyle='--')


        plt.savefig(self.outdir+f'/Validations/{self.post_ouput_str}nlines_dist{self.label}.pdf',format='pdf',dpi= 300,bbox_inches="tight")
        

    
    def plot_Normal_dist(self):
        x,psd_data= asd_data_manipulations.get_GW_data(self.config,self.gw_event_start_time,return_raw_data=True)
        x_welch,y_welch = asd_data_manipulations.get_GW_data_asd_welch(psd_end_time=self.gw_event_start_time,det=self.config['det'],
                                                    duration=self.config['duration'],                                                    
                                                    f_i=self.config['minimum_frequency'],
                                                    f_f=self.config['maximum_frequency'],
                                                    method ='welch')
                     
        x_gwtc,asd_gwtc = self.load_GWTC_curve()
        ander_gwtc=0
        I_keep= np.isin(x,x) # initiate the array 
        pval_real_gwtc=None
        if x_gwtc!=None:
            I_keep= np.isin(x,x_gwtc)

            is_it_normal_gwtc = np.real(psd_data[I_keep])/asd_gwtc
            is_it_normal_gwtc_complex = np.imag(psd_data[I_keep])/asd_gwtc
            is_it_normal_gwtc = np.concatenate([is_it_normal_gwtc, is_it_normal_gwtc_complex])
            
            pval_real_gwtc = stats.kstest(is_it_normal_gwtc,stats.norm.cdf).pvalue
            res = anderson(is_it_normal_gwtc)
            ander_gwtc = res.statistic
        #####          
        
        is_it_normal = np.real(psd_data)/self.curve_two_phases
        is_it_normal_complex  = np.imag(psd_data)/self.curve_two_phases
        is_it_normal = np.concatenate([is_it_normal[I_keep], is_it_normal_complex[I_keep]])

        res = anderson(is_it_normal)
        ander_tbilby = res.statistic
        pval_real = stats.kstest(is_it_normal,stats.norm.cdf).pvalue
        ######

        is_it_normal_welch = np.real(psd_data[I_keep])/y_welch[I_keep]
        is_it_normal_welch_complex = np.imag(psd_data[I_keep])/y_welch[I_keep]
        is_it_normal_welch = np.concatenate([is_it_normal_welch, is_it_normal_welch_complex])
        
        res = anderson(is_it_normal_welch)
        ander_welch = res.statistic
        pval_real_welch = stats.kstest(is_it_normal_welch,stats.norm.cdf).pvalue
        
        
        logger('p val tpowerbilby= ' + str(round(pval_real,2)) +  ' p val welch = ' + str(round(pval_real_welch,2))
              + 'p val GWTC = '  +str(pval_real_gwtc) , inspect.currentframe().f_code.co_name)
        logger('Anderson tpowerbilby= ' + str(round(ander_tbilby,2)) +  ' Anderson welch = ' + str(round(ander_welch,2))
              + 'Anderson GWTC = '  +str(ander_gwtc) , inspect.currentframe().f_code.co_name)
       
        F=1.5
        F= 15
        sz_F=1

        mu=0
        sigma=1
        bins = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)
        x_guass = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)

        plt.figure(figsize=(sz_F*6.18, sz_F*3.8))

        n,bins,patches = plt.hist(is_it_normal_welch,bins=bins,log=True,alpha=0.5,color='#C2A28F',label='Welch',edgecolor ='none',histtype='stepfilled') # used to be k            
        n,bins,patches = plt.hist(is_it_normal,bins,log=True,color='m',alpha=0.99,label='tPowerBilby',histtype='step') # used to be k    
        if x_gwtc!=None:
            n,bins,patches = plt.hist(is_it_normal_gwtc,bins,log=True,color='c',alpha=0.99,label='GWTC',histtype='step') # used to be red    
        plt.semilogy(x_guass, (bins[1]-bins[0])*len(is_it_normal)*stats.norm.pdf(x_guass, mu, sigma),color='k')

        plt.grid(True,linestyle='--')
            
        plt.xlabel('Whitened data',fontsize=F,fontfamily='Times New Roman')
             
        plt.legend(framealpha=0.0,frameon=False, ncol=1,loc="lower center",fontsize=F,prop={'family': 'Times New Roman'})
        ax= plt.gca()
        ax.tick_params(axis='both', labelsize=F) 
      
        output_fname=self.outdir+f'/Validations/{self.post_ouput_str}tbilby_normal_dist_{self.label}.pdf'     
        plt.savefig(output_fname,format='pdf',dpi= 300,bbox_inches="tight")
           
         
    def _log_l_arr(self,data,asd_est):
        
        arr= -0.5* (4/self.config['duration'])*(data/asd_est)**2 -2*np.log(asd_est)    
        return arr  



    def simple_plot(self):    

      

        x,y= asd_data_manipulations.get_GW_data(self.config,self.gw_event_start_time)
               
        x_welch,y_welch = asd_data_manipulations.get_GW_data_asd_welch(psd_end_time=self.gw_event_start_time,det=self.config['det'],
                                                    duration=self.config['duration'],                                                    
                                                    f_i=self.config['minimum_frequency'],
                                                    f_f=self.config['maximum_frequency'],
                                                    method ='welch')
        
        fig = plt.figure()
     
        plt.loglog(x,y,'c',label='data')
        plt.loglog(x[~self.Ikeep],y[~self.Ikeep],'or',alpha=0.4,label='Low Quality')
        
        
        plt.loglog(x_welch,y_welch,'-r',alpha=0.6,label='welch')
              
        

        with open(self.pkl_file, 'rb') as file:
            sampled_psds = pickle.load(file)
            median_curve = np.median(sampled_psds,axis=0)
        


        fig.set_size_inches(10, 6) 
        for k in np.arange(10):
             plt.loglog(x,sampled_psds[k,:],'b',alpha=0.15,label = '_Hidden label')


        plt.loglog(x,self.curve_two_phases,'-k',label='curve')


        plt.ylabel('Amplitude spectral density $[1/\\sqrt{\mathrm{Hz}}]$',fontsize=10,fontfamily='Times New Roman')
        plt.xlabel('freq. [Hz]',fontsize=10,fontfamily='Times New Roman')
        plt.loglog(x[~self.I_2nd],y[~self.I_2nd],'om',label='welch')
        plt.legend(prop={'family': 'Times New Roman'})
        plt.savefig(self.outdir+f'/Validations/{self.post_ouput_str}simple_plot_{self.label}.png')

    
        if self.PE_file is not None:            
            f = h5py.File(self.PE_file, 'r')
            PE_psds_dict = self._get_psd_from_ligo(f)
       

        
        plt.close('all')
        F=1.5
        F= int(F*12)+2
        sz_F=3
        fig = plt.figure(figsize=(sz_F*6.18, sz_F*3.8)) # golden ratio size
        plt.loglog(x,y,'k',label='Data')
        plt.loglog(x_welch,y_welch,alpha=0.8,color='#C2A28F',label='$\sigma_{\mathrm{Welch}}$')                 
        plt.loglog(x,self.curve_two_phases,'-m',label='$\sigma_{\mathrm{tPowerBilby}}$',lw=2)
        y_min, y_max = plt.ylim()
        if self.PE_file is not None:    
            plt.loglog(PE_psds_dict[self.det][0],np.sqrt(PE_psds_dict[self.det][1]),color='c',label='$\sigma_{\mathrm{GWTC}}$',lw=2,alpha=0.6)    
                
        plt.ylim(y_min, y_max)
        
        plt.ylabel('Amplitude spectral density $[1/\\sqrt{\mathrm{Hz}}]$',fontsize=F,fontfamily='Times New Roman')
        plt.xlabel('Freq. [Hz]',fontsize=F,fontfamily='Times New Roman')
         
        plt.legend(framealpha=0.0,frameon=False, ncol=1,loc="lower left",prop={'family': 'Times New Roman','size': F})
        plt.grid(True,linestyle='--')
        ax= plt.gca()
        ax.tick_params(axis='both', labelsize=F) 
        
        
        fig.savefig(self.outdir+f'/Validations/{self.post_ouput_str}{self.label}_tbilby_psd_sampling.pdf',format='pdf',dpi= 1200,bbox_inches="tight")

       
        # plot  the Log Likelihood compariosin 

            
        if self.PE_file is not None:
            
            F=1.5
            F= int(F*12)+2
            sz_F=3
            fig1 = plt.figure(figsize=(sz_F*6.18, sz_F*3.8)) # golden ratio 
            ax1= plt.subplot(1,1,1)
        
            plt.loglog(x,y,label='Data',color='k')      
            y_min, y_max = plt.ylim()      
            plt.loglog(x[~self.Ikeep],y[~self.Ikeep],'or',label='Low Quality Data',alpha=0.6)
                                  
            plt.loglog(PE_psds_dict[self.det][0],np.sqrt(PE_psds_dict[self.det][1]),color='c',label='$\sigma_{\mathrm{GWTC}}$',lw=2)    
            plt.loglog(x,self.curve_two_phases,'-m',label='$\sigma_{tPowerBilby}$',lw=2)
        
            plt.ylabel('Amplitude spectral density $[1/\\sqrt{\mathrm{Hz}}]$',fontsize=F,fontfamily='Times New Roman')        
            plt.xlabel('Freq. [Hz]',fontsize=F,fontfamily='Times New Roman')
             
            plt.legend(framealpha=0.0,frameon=False, ncol=1,loc="upper center",fontsize=F*1.5,prop={'family': 'Times New Roman','size': F})
            plt.grid(True,linestyle='--')
           
            ax= plt.gca()
            ax.tick_params(axis='both', labelsize=F) 
            plt.ylim(y_min, y_max)              
            ax1_1 = ax1.twinx()
            I_remove_PE= PE_psds_dict[self.det][1]<0.001 


            ax1_1.semilogx(x[I_remove_PE],np.cumsum(self._log_l_arr(data =y[I_remove_PE],asd_est=self.curve_two_phases[I_remove_PE]))
                            -np.cumsum(self._log_l_arr(data =y[I_remove_PE],asd_est=np.sqrt(PE_psds_dict[self.det][1])[I_remove_PE])),
                            '-b',alpha=0.5,label = ' $\\Delta \\ln \\mathcal{L}_i$' + ' All Data')
            ax1_1.semilogx(x[self.Ikeep&I_remove_PE],np.cumsum(self._log_l_arr(data =y[self.Ikeep &I_remove_PE ],asd_est=self.curve_two_phases[self.Ikeep&I_remove_PE ]))
                            -np.cumsum(self._log_l_arr(data =y[self.Ikeep&I_remove_PE ],asd_est=np.sqrt(PE_psds_dict[self.det][1])[self.Ikeep&I_remove_PE ] )),
                            '-m',alpha=0.5,label=' $\\Delta \\ln \\mathcal{L}_i$'+' High Quality Data')

                
            plt.ylabel('$\\Delta \\ln \\mathcal{L}_i$',fontsize=F,fontfamily='Times New Roman')
            
                     
            ax1_1.legend(framealpha=0.0,frameon=False, ncol=1,loc="lower right",fontsize=F,prop={'family': 'Times New Roman','size': F})
            ax1_1.tick_params(axis='both', labelsize=F) 
    
            ax1_1.grid(False)
            
            fig1.savefig(self.outdir+f'/Validations/{self.post_ouput_str}{self.label}tbilby_LIGO_compare_logLikelihood.pdf',format='pdf',dpi= 1200,bbox_inches="tight")

        
            


      


        
       

        
     
        
            
        
       