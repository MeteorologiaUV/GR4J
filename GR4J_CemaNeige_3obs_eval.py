# -*- coding: utf-8 -*-
"""
This file runs GR4J in CAMELS-CL dataset

@author: Luis De la Fuente
"""

#%% libraries
import pandas as pd
from tqdm import tqdm #progress bar
import numpy as np
import math
from typing import Callable
import scipy.stats
import pickle #to save
import os
import random as rand
from datetime import timedelta
import sys
import argparse


#%% Initialization
global num_calibrations
num_calibrations = 22
global path_data
path_data = '/home/u16/ldelafue/Documents/GR4J_project/Variables3.csv'
global path_par
path_par = '/home/u16/ldelafue/Documents/GR4J_project/results/3PP_model_KGE'

#%% Functions
def RMSE_metric(obs, sim):
    sim = sim[obs.notna()]
    obs = obs.dropna()
    rmse = np.mean((sim - obs)**2)**0.5
    return rmse

def NSE_metric(obs, sim, mean):
    sim = sim[obs.notna()]
    obs = obs.dropna()
    nse = 1 - np.mean((sim - obs)**2) / np.mean((sim - mean)**2)
    return nse


def KGE_metric(Sim,Obs):
    Sim = Sim[Obs.notna()]
    Obs = Obs.dropna()
    mean_s=Sim.mean()
    mean_o=Obs.mean()  
    std_s=Sim.std()
    std_o=Obs.std()
    r, _ = scipy.stats.pearsonr(Sim, Obs)
    KGE = 1 - ((r-1)**2 + (std_s/std_o-1)**2 + (mean_s/mean_o-1)**2)**0.5
    return KGE

def Cemaneige(df, theta1: float, theta2:float, alfa_PP1:float, alfa_PP2:float, alfa_PP3:float ):
    df_i = df.copy()
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
        if df_i.loc[i,'Tmax-0'] <= T_threshold_min:
            df_i.at[i,'%snow'] = 1
        else:
            if df_i.loc[i,'Tmin-0'] >= T_threshold_max:
                df_i.at[i,'%snow'] = 0
            else:
                if (df_i.loc[i,'Tmax-0'] - T_threshold_min)<T_range[i]:
                    df_i.at[i,'%snow'] = 1 - (df_i.loc[i,'Tmax-0'] - T_threshold_min)/T_range[i]
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
            df_i.at[i,'Snow_T'] = min(0,theta2 * df_i.Snow_T[i - timedelta(days=1)] + (1 - theta2) * df_i.loc[i,'Tmean-0'] )   

                                      
        if df_i.Snow_T[i] == 0:
            df_i.at[i,'Pot_melt'] =  min(df_i.Snow_bef_melt[i],max(0,theta1 * df_i.loc[i,'Tmean-0']))
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
        
        

    
def GR4J_local(df, alfa_PP1:float, alfa_PP2:float, alfa_PP3:float, theta1: float, theta2:float, x1: float, x2:float, x3:float, x4:float):

    
    [rows, cols] = df.shape

    df_i = Cemaneige(df, theta1, theta2, alfa_PP1, alfa_PP2, alfa_PP3)       
    df_i.loc[:,'Pn'] = df_i['P'] - df_i['PET']
    df_i.loc[:,'En'] = 0.0
    df_i.loc[df_i.Pn<0,'En'] =  df_i['PET'] - df_i['P'] 
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
    pred = df_i.Qpred
    obs = df_i.caudal_mean
    
    return pred, obs, df_i

    
def spliting(df, sim, obs, period):
    """
    O: Calibration
    1: Evaluacion
    2: Testeo
    3: Menos de 12 anos
    """
    if period == 'Calibration':
        index = 0
    elif period == 'Evaluation':
        index = 1
    elif period == 'Testing':
        index = 2
    else:
        index =3
    index_list = df.caudal_mask2 == index
    sim = sim[index_list]
    obs = obs[index_list]

    #NaN_index = obs == np.NAN
    #print(NaN_index)
    #sim = sim[~NaN_index]
    #obs = obs[~NaN_index]    
    #print(sim)
    #print(obs)
    
    if len(sim) == 0:
        print('---------------------------------------------------------')
        print(f'The catchment selected is not in the {period} period')
        print('---------------------------------------------------------')
        print(index)
    return sim, obs, df        

    
def load_data(ID):
    data = pd.read_csv(path_data)
    data = data[data.gauge_id == ID] #self.code
    #print(data)
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

    
    return df


def GR4J_eval(ID, period:str):

    parameters = np.zeros((num_calibrations,11)) # OF + n° parameters
    parameters = pd.DataFrame(parameters, columns=['alfa_PP1','alfa_PP2','alfa_PP3','theta1', 'theta2', 'x1', 'x2', 'x3', 'x4','KGE','iter']) 
    
    
    for j in tqdm(range(num_calibrations)):
    
        full_path = path_par + '/' + str(ID) + '_iter' + str(j) + '.par'
        df_p = pd.read_csv(full_path)
        df_p.drop(columns=['code.1'], inplace=True)

        parameters.at[j,'alfa_PP1'] = df_p.iloc[0,1]
        parameters.at[j,'alfa_PP2'] = df_p.iloc[0,2]
        parameters.at[j,'alfa_PP3'] = df_p.iloc[0,3]
        parameters.at[j,'theta1'] = df_p.iloc[0,4]
        parameters.at[j,'theta2'] = df_p.iloc[0,5]
        parameters.at[j,'x1'] = df_p.iloc[0,6]
        parameters.at[j,'x2'] = df_p.iloc[0,7]
        parameters.at[j,'x3'] = df_p.iloc[0,8]
        parameters.at[j,'x4'] = df_p.iloc[0,9]
        parameters.at[j,'iter'] = j
        
        #parameters.at[j,11] = df_p.iloc[0,11]

        
        df = load_data(ID)
        sim, obs, df = GR4J_local(df, df_p.alfa_PP1[0], df_p.alfa_PP2[0], df_p.alfa_PP3[0], df_p.theta1[0], df_p.theta2[0], df_p.x1[0], df_p.x2[0], df_p.x3[0], df_p.x4[0])
        sim, obs, df = spliting(df, sim, obs, period) 
        parameters.at[j,'KGE'] = KGE_metric(sim,obs)


    
    parameters.sort_values(by=['KGE'], ascending=False, inplace=True)
    filepath = path_par + '/' + str(ID) + '_eval.csv'
    parameters.to_csv(filepath)
    print(parameters)
    
    return
   
#%% Main
parser = argparse.ArgumentParser()
parser.add_argument('--catch', type=int)
parser.add_argument('--period', type=str)
cfg = vars(parser.parse_args()) 
ID = cfg["catch"]
period = cfg["period"]
parameters = GR4J_eval(ID,period)

    
