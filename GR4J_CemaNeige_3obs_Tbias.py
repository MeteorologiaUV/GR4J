#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:54:03 2020

@author: ldelafue
"""

import spotpy
import numpy as np
import pandas as pd
import math
from typing import Callable
from pyDOE import * #hipercube
import scipy.stats
#import pickle #to save
#import feather #to save
import random as rand
import os
from datetime import timedelta
import argparse
from sys import exit



class spotpy_setup():
    def __init__(self,c,dim=10):
        self.dim = dim
        self.catchment = c
        self.params = [spotpy.parameter.Uniform('a1',low=0, high=2.0),
                       spotpy.parameter.Uniform('a2',low=0, high=2.0),
                       spotpy.parameter.Uniform('a3',low=0, high=2.0),
                       spotpy.parameter.Uniform('Tg1',low=0.1, high=7), #degree-day factor
                       spotpy.parameter.Uniform('Tg2',low=0, high=1.0), #x6: snowpack inertia factor
                       spotpy.parameter.Uniform('x1',low=1, high=5000), #x1: Capacity of production store (mm)
                       spotpy.parameter.Uniform('x2',low=-10, high=10), #x2: Water exchange coefficient (mm)
                       spotpy.parameter.Uniform('x3',low=1, high=1500), #x3: Capacity of routing store (mm)
                       spotpy.parameter.Uniform('x4',low=0.501, high=4.5), #x4: UH time base (days)
                       spotpy.parameter.Uniform('Tbias',low=-3, high=3), #Temperature bias
                       
                       ]
        self.model = model(self.catchment)
        self.sim = None
        self.obs =None
        
    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    def simulation(self, vector):

        sim, obs = self.model.GR4J_local(alfa_PP1 =vector[0], alfa_PP2 =vector[1], alfa_PP3 =vector[2], theta1 = vector[3], theta2 = vector[4], x1=vector[5], x2=vector[6], x3=vector[7], x4=vector[8], Tbias=vector[9])

        sim, obs = self.model.spliting(sim, obs, 'Calibration')

        # self.sim = sim[obs.notna()]
        # self.obs = obs.dropna()
        self.sim = sim[warm_up:]
        self.obs = obs[warm_up:]
        
        return self.sim
    
    def evaluation(self):
        observations = self.obs
        return observations
            

    def objectivefunction(self, simulation, evaluation):

        if it<20:
            like = spotpy.objectivefunctions.kge(self.evaluation(),simulation) #possitive for maximization algorithms  
        else:
            like = -spotpy.objectivefunctions.kge(self.evaluation(),simulation) #negative for minimization algorithms  
   
        return like


class model():
    def __init__(self,c_ID):

        self.path = current_path
        self.code = c_ID
        self.df = None
        self.pred = None
        self.obs = None
        self.l_ini = None        
        self.load_data()
        

        
    def load_data(self):
        dataset_path = self.path + '/Variables3.csv'
        data = pd.read_csv(dataset_path)
        data = data[data.gauge_id == self.code] #self.code
        
        df = data[{'date','gauge_id','caudal_mask2',
                    'pp_o_era5_pp_mean_b_none_d1_p0d',
                    'pp_o_imerg_pp_mean_b_none_d1_p0d',
                    'pp_o_pdir_pp_mean_b_none_d1_p0d',
                    'tmp_o_era5_tmax_mean_b_none_d1_p0d',
                    'tmp_o_era5_tmp_mean_b_none_d1_p0d',
                    'tmp_o_era5_tmin_mean_b_none_d1_p0d',
                    'caudal_mean',
                    'top_s_cam_elev_mean_b_none_c_c',
                    'top_s_cam_area_tot_b_none_c_c',
                    'X',
                    'Y'}]
        df['date'] = pd.to_datetime(df['date'])
        df.index = df['date']
        df = df.drop(['date'], axis=1)
        df['PP_ERA5'] =df.pp_o_era5_pp_mean_b_none_d1_p0d / 100
        df['PP_IMERG'] =df.pp_o_imerg_pp_mean_b_none_d1_p0d / 10
        df['PP_PERSIANN'] = df.pp_o_pdir_pp_mean_b_none_d1_p0d / 10
        df['Tmax-0'] = df.tmp_o_era5_tmax_mean_b_none_d1_p0d / 10
        df['Tmean-0'] = df.tmp_o_era5_tmp_mean_b_none_d1_p0d / 10
        df['Tmin-0'] = df.tmp_o_era5_tmin_mean_b_none_d1_p0d / 10
        df['caudal_mean'] = df.caudal_mean / 10
        df['elev_mean'] = df.top_s_cam_elev_mean_b_none_c_c / 1
        df['area'] = df.top_s_cam_area_tot_b_none_c_c / 10
        df['gauge_lat'] = df.Y
        df['gauge_lon'] = df.X
        #df['gauge_lat'] = df.top_s_cam_lat_none_p_none_c_c
        #df['gauge_lon'] = df.top_s_cam_lon_none_p_none_c_c

        df.caudal_mean = 86.4*df.caudal_mean/df.area

        lat = df.loc[df.index[0],'gauge_lat']*2*np.pi/360

        jul = pd.to_datetime(df.index.year.map(str) + '/01/01', format="%Y/%m/%d")
        df['PET'] = 0.408*0.0023*(df['Tmax-0'] - df['Tmin-0'])**0.5*(0.5*df['Tmax-0'] + 0.5*df['Tmin-0'] + 17.8)
        df['julian'] = df.index.to_julian_date() - jul.to_julian_date() + 1
        df['gamma'] = 0.4093*np.sin(2*np.pi*df.julian/365 - 1.405)
        df['hs'] = np.arccos(-np.tan(lat)*np.tan(df.gamma))
        df.PET = 3.7595*10*(df.hs*np.sin(lat)*np.sin(df.gamma)+np.cos(lat)*np.cos(df.gamma)*np.sin(df.hs))*df.PET

        
        self.df = df
        

        return
        
    def spliting(self, sim, obs, period):
        """
        O: Calibration
        1: Evaluacion
        2: Testeo
        3: Menos de 12 anos
        """
        if period == 'Calibration':
            index = 0
        elif period == 'Evaluation':
            index == 1
        elif period == 'Testeo':
            index == 2
        else:
            index == 3
        index_list = self.df.caudal_mask2 == index
        sim = sim[index_list]
        obs = obs[index_list]
        if len(sim) == 0:
            print('---------------------------------------------------------')
            print(f'The catchment selected is not in the {period} period')
            print('---------------------------------------------------------')
            exit(0)
        return sim, obs
        
    def Cemaneige(self, theta1: float, theta2:float, alfa_PP1:float, alfa_PP2:float, alfa_PP3:float, Tbias:float ):
        lat = self.df.loc[self.df.index[0],'gauge_lat']*2*np.pi/360 
        self.df['PET_bias'] = 0.408*0.0023*(self.df['Tmax-0'] - self.df['Tmin-0'])**0.5*(0.5*self.df['Tmax-0'] + 0.5*self.df['Tmin-0'] + 17.8 + Tbias)
        self.df['PET_bias']  = 3.7595*10*(self.df.hs*np.sin(lat)*np.sin(self.df.gamma)+np.cos(lat)*np.cos(self.df.gamma)*np.sin(self.df.hs))*self.df.PET_bias



        
        df_i = self.df
        alfa_IMERG = alfa_PP1
        alfa_PERSIANS = alfa_PP2
        alfa_ERA5 = alfa_PP3
        df_i.loc[:,'P_total'] = alfa_IMERG * df_i.loc[:,'PP_IMERG'] + alfa_PERSIANS * df_i.loc[:,'PP_PERSIANN'] + alfa_ERA5 * df_i.loc[:,'PP_ERA5']
        #print(df_i.isna().sum())
        if df_i.elev_mean[0] < 1500:
            T_threshold_max = 0.0
            T_threshold_min= 0.0
            T_range = df_i.loc[:,'Tmax-0'] - df_i.loc[:,'Tmin-0'] 
        else:
            T_threshold_max = 3.0
            T_threshold_min= -1.0
            T_range = df_i.loc[:,'Tmax-0'] * 0.0 + 4 
        df_i.loc[:,'%snow'] = 0.0
        df_i.loc[:,'PP_liq'] = 0.0
        df_i.loc[:,'PP_sol'] = 0.0
        df_i.loc[:,'Snow_bef_melt'] = 0.0
        df_i.loc[:,'Snow_T'] = 0.0      
        df_i.loc[:,'Pot_melt'] = 0.0
        df_i.loc[:,'%cover'] = 0.0
        df_i.loc[:,'melt'] = 0.0
        df_i.loc[:,'Snow'] = 0.0        
        df_i.loc[:,'P'] = 0.0    
 
        for i in df_i.index:
            if df_i.loc[i,'Tmax-0'] + Tbias <= T_threshold_min:
                df_i.at[i,'%snow'] = 1
            else:
                if df_i.loc[i,'Tmin-0'] + Tbias >= T_threshold_max:
                    df_i.at[i,'%snow'] = 0
                else:
                    if (df_i.loc[i,'Tmax-0'] + Tbias - T_threshold_min)<T_range[i]:
                        df_i.at[i,'%snow'] = 1 - (df_i.loc[i,'Tmax-0'] + Tbias - T_threshold_min)/T_range[i]
                    else:
                        df_i.at[i,'%snow'] = 0
            df_i.at[i,'PP_liq'] = df_i.loc[i,'P_total'] * (1 - df_i.loc[i,'%snow'])
            df_i.at[i,'PP_sol'] = df_i.loc[i,'P_total'] * df_i.loc[i,'%snow']                     
                    
        Snow_threshold =   df_i.PP_sol.mean() * 365.25 * 0.9
                  
        for i in df_i.index:                 
            if i == df_i.index[0]:
                df_i.at[i,'Snow_bef_melt'] = 0
                df_i.at[i,'Snow_T'] = 0
            else:
                df_i.at[i,'Snow_bef_melt'] = df_i.PP_sol[i] + df_i.Snow[i - timedelta(days=1)]
                df_i.at[i,'Snow_T'] = min(0,theta2 * df_i.Snow_T[i - timedelta(days=1)] + (1 - theta2) * (df_i.loc[i,'Tmean-0'] + Tbias) )      

                                          
            if df_i.Snow_T[i] == 0:
                df_i.at[i,'Pot_melt'] =  min(df_i.Snow_bef_melt[i],max(0,theta1 * (df_i.loc[i,'Tmean-0'] + Tbias)))
            else:
                df_i.at[i,'Pot_melt'] = 0
            if df_i.Snow_bef_melt[i] < Snow_threshold:
                df_i.at[i,'%cover'] = df_i.Snow_bef_melt[i] / Snow_threshold
            else:
                df_i.at[i,'%cover'] = 1
            df_i.at[i,'melt'] = (0.9* df_i.loc[i,'%cover'] + 0.1) * df_i.loc[i,'Pot_melt']
            df_i.at[i,'Snow'] = df_i.loc[i,'Snow_bef_melt'] - df_i.loc[i,'melt']
            df_i.at[i, 'P'] = df_i.loc[i,'melt'] + df_i.loc[i,'PP_liq']


        return df_i
            
            

        
    def GR4J_local(self, alfa_PP1:float, alfa_PP2:float, alfa_PP3:float, theta1: float, theta2:float, x1: float, x2:float, x3:float, x4:float, Tbias:float):

        
        [rows, cols] = self.df.shape

        df_i = self.Cemaneige(theta1, theta2, alfa_PP1, alfa_PP2, alfa_PP3, Tbias)       
        df_i.loc[:,'Pn'] = df_i['P'] - df_i['PET_bias']
        df_i.loc[:,'En'] = 0.0
        df_i.loc[df_i.Pn<0,'En'] =  df_i['PET_bias'] - df_i['P'] 
        df_i.loc[df_i.Pn<0, 'Pn'] = 0.0
        df_i.loc[:,'S'] = 150.0 
        df_i.loc[:,'Ps'] = 0.0
        df_i.loc[:,'Es'] = 0.0    
        S_initial = 150
        for i in df_i.index:
            if i == df_i.index[0]:
                df_i.at[i,'Ps'] = (x1*(1-(S_initial/x1)**2)*np.tanh(df_i.Pn[i]/x1))/(1+(S_initial/x1)*np.tanh(df_i.Pn[i]/x1))
                df_i.at[i,'Es'] = (S_initial*(2-(S_initial/x1))*np.tanh(df_i.En[i]/x1))/(1+(1-S_initial/x1)*np.tanh(df_i.En[i]/x1))
                df_i.at[i,'S'] = S_initial - df_i.Es[i] + df_i.Ps[i]
                df_i.at[i,'Perc'] = df_i.S[i]*(1-(1+(4*df_i.S[i]/(9*x1))**4)**(-0.25))
                df_i.at[i,'S'] = df_i.S[i] - df_i.Perc[i]
                prev = i
            else:
                df_i.at[i,'Ps'] = (x1*(1-(df_i.S[prev]/x1)**2)*np.tanh(df_i.Pn[i]/x1))/(1+(df_i.S[prev]/x1)*np.tanh(df_i.Pn[i]/x1))
                df_i.at[i,'Es']  = (df_i.S[prev]*(2-(df_i.S[prev]/x1))*np.tanh(df_i.En[i]/x1))/(1+(1-df_i.S[prev]/x1)*np.tanh(df_i.En[i]/x1))
                df_i.at[i,'S'] = df_i.S[prev] - df_i.Es[i] + df_i.Ps[i]
                df_i.at[i,'Perc'] = df_i.S[i]*(1-(1+(4*df_i.S[i]/(9*x1))**4)**(-0.25))
                df_i.at[i,'S'] = df_i.S[i] - df_i.Perc[i]
                prev = i
        df_i.loc[:,'Pr'] = df_i.Perc + df_i.Pn - df_i.Ps
        n = math.ceil(x4)
        m = math.ceil(2*x4)
        tn = np.array(list(range(n+1)))
        tm = np.array(list(range(m+1)))
        SH1 = np.zeros(n+1)
        SH2 = np.zeros(m+1)
        for i in range(n+1):
            if i<x4:
                SH1[i] = (i/x4)**(5/2)
            else:
                SH1[i] = 1
    
        for i in range(m+1):
            if i <= x4:
                SH2[i] = 0.5*(i/x4)**(5/2)
            elif i< 2*x4:
                SH2[i] = 1-0.5*(2-i/x4)**(5/2)
            else:
                SH2[i] = 1
        UH1 = np.zeros(n)
        for i in range(1,n+1,1):
            UH1[i-1] = SH1[i] - SH1[i-1]   
        UH2 = np.zeros(m)
        for i in range(1,m+1,1):
            UH2[i-1] = SH2[i] - SH2[i-1]
        R_initial = 45
        df_i.loc[:,'F'] = 0.0
        df_i.loc[:,'Q9'] = 0.0
        df_i.loc[:,'Q1'] = 0.0
        df_i.loc[:,'Qr'] = 0.0   
        df_i.loc[:,'Qd'] = 0.0
        df_i.loc[:,'Qpred'] = 0.0  
        conv_1 = np.zeros((rows,rows+m))
        conv_1 = pd.DataFrame(conv_1)
        conv_1.index = df_i.index
        conv_2 = np.zeros((rows,rows+m))
        conv_2 = pd.DataFrame(conv_2)
        conv_2.index = df_i.index
        j = 0
        for i in df_i.index:
            conv_2.loc[i,j:j+m-1] = 0.1*df_i.Pr[i]*UH2.T
            df_i.at[i,'Q1'] = conv_2[j].sum()
            if i == df_i.index[0]:
                df_i.at[i,'F'] = x2*(R_initial/x3)**(7/2) #F possitive: river and groundwater gain water
                conv_1.loc[i,j:j+n-1] = 0.9*df_i.Pr[i]*UH1.T
                df_i.at[i,'Q9'] = conv_1[j].sum()
                df_i.at[i,'R'] = R_initial + df_i.Q9[i] + df_i.F[i] 
            else:
                df_i.at[i,'F'] = x2*(df_i.R[prev]/x3)**(7/2)
                conv_1.loc[i,j:j+n-1] = 0.9*df_i.Pr[i]*UH1.T
                df_i.at[i,'Q9'] = conv_1[j].sum()
                df_i.at[i,'R'] = df_i.R[prev] + df_i.Q9[i] + df_i.F[i] 
    
            if df_i.R[i]<0:
                df_i.at[i,'R'] = 0
            df_i.at[i,'Qr'] = df_i.R[i]*(1-(1+(df_i.R[i]/x3)**4)**(-0.25))
            df_i.at[i,'R'] = df_i.R[i] - df_i.Qr[i]
            df_i.at[i,'Qd'] = df_i.Q1[i] + df_i.F[i] 
            if df_i.Qd[i]<0:
                df_i.at[i,'Qd'] = 0
            df_i.at[i,'Qpred'] = df_i.Qr[i] + df_i.Qd[i]
            j = j + 1
            prev = i
        self.df =  df_i
        self.pred = df_i.Qpred
        self.obs = df_i.caudal_mean
        
        return self.pred, self.obs
        
def GR4J_snow_bias(c:int):
    
    iterations = 20
    global current_path
    current_path = os.getcwd()
    global it    
    global warm_up
    warm_up = 365
    global spotpy_setup
    spotpy_setup = spotpy_setup(c)  

    for it in range(iterations):
        print(it)
        if it<20:
            n_sample = 800
        else:
            n_sample = 5000
        
        if it<10:
            sampler = spotpy.algorithms.mle(spotpy_setup, dbname=str(c_ID)+ "_KGE", dbformat='csv') 
        elif it >=10 and it < 20:
            sampler = spotpy.algorithms.demcz(spotpy_setup, dbname=str(c_ID)+ "_KGE", dbformat='csv')    
        else:
            sampler = spotpy.algorithms.sceua(spotpy_setup, dbname=str(c_ID)+ "_KGE", dbformat='csv')

        if it<20:
            sampler.sample(n_sample)
            results=sampler.getdata()
            best = spotpy.analyser.get_best_parameterset(results, maximize=True) # the best result
        else:
            sampler.sample(n_sample, ngs=10) 
            results=sampler.getdata()
            best = spotpy.analyser.get_best_parameterset(results, maximize=False) # the best result


        index_best = spotpy.analyser.get_maxlikeindex(results)
        best = best[0]

        
        parameters = np.zeros((1,12)) # OF + n° parameters
        parameters = pd.DataFrame(parameters, columns=['alfa_PP1','alfa_PP2','alfa_PP3','theta1', 'theta2', 'x1', 'x2', 'x3', 'x4', 'Tbias','code','KGE_train']) 
        parameters.at[0,'code'] = c_ID
        parameters.at[0,'alfa_PP1'] = best[0] 
        parameters.at[0,'alfa_PP2'] = best[1] 
        parameters.at[0,'alfa_PP3'] = best[2] 
        parameters.at[0,'theta1'] = best[3] 
        parameters.at[0,'theta2'] = best[4] 
        parameters.at[0,'x1'] = best[5] 
        parameters.at[0,'x2'] = best[6] 
        parameters.at[0,'x3'] = best[7] 
        parameters.at[0,'x4'] = best[8]
        parameters.at[0,'Tbias'] = best[9] 
        parameters.at[0,'KGE_train'] = index_best[1] 
        parameters.index = parameters.code
        print(parameters)
        
        filepath = current_path + '/results/3PP_model_KGE/' + str(c) + '_iter_b_' + str(it) + '.par' #CHECK FOLDER
        parameters.to_csv(filepath)
        
    return

parser = argparse.ArgumentParser()
parser.add_argument('--catch', type=int)
cfg = vars(parser.parse_args()) 
print(cfg["catch"])
c_ID = cfg["catch"]
GR4J_snow_bias(c_ID)
