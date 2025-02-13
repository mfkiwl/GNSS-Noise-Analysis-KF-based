# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:46:24 2025

@author: cdrg
"""

import os
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import lombscargle
import gc

# site
from numba import jit
from statsmodels.tsa.arima.model import ARIMA

# local
from kf import kf_main, rts, mle_apso


#=====================================================================
# Dir
GPS_DIR = "D:/RA_all/taiwan_all/output/old_format/"
OFFSET_DIR = "D:/RA_all/GS_stations_noise/data/gps_timeseries/IGb14/"
POST_DIR = "D:/RA_all/wls_post/output/post_const/"
OUTPUT_DIR = "D:/RA_all/taiwan_all_ls_post/output/kf_ar3/"

gps_list = os.listdir(GPS_DIR)
#gps_list = gps_list[gps_list.index("SUCH.txt")::]
#=====================================================================
# Init
AR_ORDER = 3
EPS = 1e-9

# PSO
p_num = 40
itr = 40
theta_num = 5
lb = np.full(theta_num, EPS)
lb[-1] = 0.1
lb[-2] = 0.1
ub = np.array([2, 0.008, 0.002, 4, 4]) #E-W
#ub = np.array([2, 0.04, 0.008, 4, 4])   #U-D

#=====================================================================
# Component {E-W : 1, N-S : 2, U-D : 3}
component = 1
if component == 1 : 
    component_str = 'E-W'
    POST_DIR = POST_DIR + 'E-W/'
    OUTPUT_DIR = OUTPUT_DIR + 'E-W/'
    
if component == 2 : 
    component_str = 'N-S'
    POST_DIR = POST_DIR + 'N-S/'
    OUTPUT_DIR = OUTPUT_DIR + 'N-S/'
    
if component == 3 : 
    component_str = 'U-D'
    POST_DIR = POST_DIR + 'U-D/'
    OUTPUT_DIR = OUTPUT_DIR + 'U-D/'

#=====================================================================
# Function
def import_data(data_path, filename):
    data = []
    with open(data_path + filename) as file:  
        for line in file :
            data.append(line.rstrip().split())
    return data


def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def date2decy(year, month, day, hour, minute, second):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap_year(year) : days_in_month[1] = 29
    
    day_of_year = sum(days_in_month[:month - 1]) + day - 1
    seconds_in_day = hour * 3600 + minute * 60 + second
    
    total_seconds_in_year = (366 if is_leap_year(year) else 365) * 24 * 3600
    current_seconds = day_of_year * 86400 + seconds_in_day
    
    return round(year + current_seconds / total_seconds_in_year, 5)


@jit(nopython=True)
def get_G_base(time):
    t_dim = time.shape[0]
    
    G = np.zeros((t_dim, 6))
    for i in range(t_dim):
        G[i, 0] = 1
        G[i, 1] = time[i] - time[0]
        G[i, 2] = np.sin(2 * np.pi * time[i])
        G[i, 3] = np.cos(2 * np.pi * time[i])
        G[i, 4] = np.sin(4 * np.pi * time[i])
        G[i, 5] = np.cos(4 * np.pi * time[i])
   
    return G


@jit(nopython=True)
def get_G_offset(time, offset_time):
    t_dim = time.shape[0]
    offset_dim = offset_time.shape[0]
    
    G = np.zeros((t_dim, offset_dim))
    for j in range(offset_dim):
       G[:, j] = time>=offset_time[j]
                
    return G


@jit(nopython=True, fastmath=True)
def post(time, post_const, post_decy):
    t_dim = time.shape[0]
    post_num = len(post_const)
    post = np.zeros((t_dim, post_num))
    
    for i in range(post_num):
        T = int(post_const[i])/365.25
        post_t = time - post_decy[i]
        for t in range(t_dim):
            if post_t[t]>=0:
                post[t, i] += np.log(1 + post_t[t]/T)
        
    return post


@jit(nopython=True)
def wls(time, offset_decy, post_constant, post_decy, yw, y_sig):
    G = get_G_base(time)

    if len(offset_decy):
        G_offset = get_G_offset(time, offset_decy)
        
        if len(post_constant):
            G_post = post(time, post_constant, post_decy)
            G = np.hstack((G, G_offset, G_post))
            
        else : G = np.hstack((G, G_offset))
    
    W = np.diag(1/y_sig)
    Gw = np.dot(W, G)
    result = np.linalg.lstsq(Gw, yw)
    
    return G, result


def get_yw(y, sig):
    W = np.diag(1/sig)
    
    return W @ y


def get_Q(t_dim, x_dim):
    return np.zeros((t_dim-1, x_dim, x_dim))


def get_F(t_dim, x_dim, dt, ar_para, ar_start_ind, ar_order):
    F = np.tile(np.eye(x_dim), (t_dim-1, 1, 1))
    F[:, 0, 1] = dt
    
    for i in range(ar_order):
        if i!=0 : F[:, -i, -i] = 0
        F[:, ar_start_ind+i, ar_start_ind] = ar_para[i]
        if i < ar_order - 1:
            F[:, ar_start_ind+i, ar_start_ind + i + 1] = 1
        
    return F


@jit(nopython=True)
def get_H(t_dim, x_dim, time, ar_order, offset_decy):
    H = np.zeros((t_dim, 1, x_dim))
    offset_num = len(offset_decy)
    
    for t in range(len(time)):
        H[t, 0, 0] = 1
        H[t, 0, 2] = np.sin(2 * np.pi * time[t])
        H[t, 0, 3] = np.cos(2 * np.pi * time[t])
        H[t, 0, 4] = np.sin(4 * np.pi * time[t])
        H[t, 0, 5] = np.cos(4 * np.pi * time[t])
        
        if len(offset_decy):  
            for o in range(offset_num):
                if time[t] >= offset_decy[o]:
                    H[t, 0, 6:7+o] = 1
                    
        if ar_order:
            H[t, 0, -ar_order] = 1
       
    return H


def get_R(sig):
    return sig**2
    

#=====================================================================
v_sig_list = []
v_wls_sig_list = []
t_list = np.array(np.arange(700, 2400, 50), dtype=int)

for a in t_list:
    for st in gps_list:
        name, ext = st.split(".")
        if name not in ['WANL']:continue
        # if os.path.exists(OUTPUT_DIR + name + "/pso.txt"):
        #     continue
        if ext != "txt" : continue
        print("#=========================================")
        print(name)
        
        if not os.path.exists(OUTPUT_DIR + name) : 
            os.makedirs(OUTPUT_DIR + name)
    
        #=================================================================
        # Import data
        gps = np.array(import_data(GPS_DIR, st)[2::], dtype=float)
        gps = gps[0:a]
        
        time, gps_y, gps_y_sig = gps[:, [0, component, component+3]].T
        
        gps_y, gps_y_sig = gps_y * 1e3, gps_y_sig * 1e3
        
        #=================================================================
        # Full data
        gps_start_datetime = datetime(*map(int, gps[0, 11::]))
        gps_end_datetime = datetime(*map(int, gps[-1, 11::]))
        
        gps_full_datetime = [gps_start_datetime]
        gps_full_decy = [date2decy(*map(int, gps[0, 11::]))]
        
        # full time
        gps_full_datetime = [gps_start_datetime + timedelta(days=i)
                             for i in range((gps_end_datetime - gps_start_datetime).days + 1)]
        
        gps_full_decy = np.array([date2decy(*dt.timetuple()[:6]) for dt in gps_full_datetime])
        
        # full data
        gps_full_ind = np.isin(gps_full_decy, time)
        gps_full_len = len(gps_full_decy)
        
        gps_full_y = np.full(gps_full_len, np.nan)
        gps_full_y_sig = np.full(gps_full_len, np.nan)
        
        gps_full_y[gps_full_ind] = gps_y
        gps_full_y_sig[gps_full_ind] = gps_y_sig
        
        del ext, gps, gps_start_datetime, gps_end_datetime, gps_full_ind, gps_full_len
        
        #=================================================================
        # offset
        offset_filename = name + '.break'
        offset_decy = np.array([])
        
        if os.path.exists(OFFSET_DIR + offset_filename):
            offset_file = [list(map(int,a[1::]))
                      for a in import_data(OFFSET_DIR, offset_filename)]
            
            offset_time_datetime = [datetime(*a[0:6]) for a in offset_file]
            offset_time_datetime = [a for a in offset_time_datetime
                                    if a <= gps_full_datetime[-1] and a > gps_full_datetime[0]]
            
            offset_decy = np.array([date2decy(*a.timetuple()[:6]) for a in offset_time_datetime])
            '''
            offset_ind = [next(i for i, b in enumerate(time) if b>=a)
                          for a in offset_decy]
                
            offset_full_ind = [next(i for i, b in enumerate(gps_full_decy) if b>=a)
                          for a in offset_decy]
            '''
            del offset_file, offset_time_datetime
         
        del gps_full_datetime, offset_filename
        #=================================================================
        # Post
        post_decy = np.array([])
        post_const = np.array([])
        post_ind = np.array([])
        
        if os.path.exists(POST_DIR + st):
            post_decy, post_const = np.array(import_data(POST_DIR, st)[1::], dtype=float).T
            post_decy_ind = [i for i, a in enumerate(post_decy) if a>time[0] and a<time[-1]]
            post_decy = post_decy[post_decy_ind]
            post_const = post_const[post_decy_ind]
            post_ind = np.array([next(i for i, b in enumerate(gps_full_decy) if b>=a)
                          for a in post_decy])
            
        #=================================================================
        # WLS
        yw = get_yw(gps_y, gps_y_sig)
        
        G, wls_result = wls(time, offset_decy, post_const, post_decy, yw, gps_y_sig)
        
        # full G
        G_full = get_G_base(gps_full_decy)
        
        if len(offset_decy): 
            G_offset = get_G_offset(gps_full_decy, offset_decy)
               
            if len(post_const): 
                G_post = post(gps_full_decy, post_const, post_decy)
                G_full = np.hstack((G_full, G_offset, G_post))
                
                del G_post
                
            else:
                G_full = np.hstack((G_full, G_offset))
                
            del G_offset
         
        wls_m = wls_result[0]  
         
        wls_fit = G_full @ wls_m 
        wls_res = gps_full_y - wls_fit
        
        gtg_inv = np.linalg.inv(G.T @ G)
        sigma2 = np.sum((G@wls_m - gps_y)**2) / (G.shape[0] - G.shape[1])
        C_x = sigma2 * gtg_inv
        v_wls_sig_list.append(np.sqrt(C_x[1, 1]))
        # component
        wls_trend = G_full[:, 0:2] @ wls_m [0:2]
        wls_annual = G_full[:,2:4] @ wls_m[2:4]
        wls_semi_annual = G_full[:,4:6] @ wls_m[4:6]
        
        if len(offset_decy):
            wls_offset = G_full[:,6:6+len(offset_decy)] @ wls_m [6:6+len(offset_decy)]           
            wls_trend = [a + b for a, b in zip(wls_offset, wls_trend)]
            del wls_offset
            
        if len(post_const):
            wls_post = G_full[:,6+len(offset_decy):6+len(offset_decy)+len(post_const)] @\
                wls_m [6+len(offset_decy):6+len(offset_decy)+len(post_const)]
                
            wls_trend = [a + b for a, b in zip(wls_post, wls_trend)]
            del wls_post
          
        # PSD
        freq = (365.25/len(gps_full_decy))*np.arange(1,len(gps_full_decy))
        freq_cpy = np.array([a for a in freq if a<= 365.25/2])
        freq_omega_cpy = freq_cpy * 2 * np.pi
        
        psd_lom = lombscargle(gps_full_decy[~np.isnan(wls_res)],\
                              wls_res[~np.isnan(wls_res)], freq_omega_cpy)/365.25*2
            
        #=================================================================
        # Plot
    
        
        del psd_lom, wls_annual, wls_fit, G_full, post_const, wls_result,\
            wls_semi_annual, wls_trend, yw
            
        #=================================================================
        # WLS residual ARMA
        ARIMA_model = ARIMA(wls_res, order=(AR_ORDER, 0, 0))   
        ARIMA_results = ARIMA_model.fit(method = "statespace")
        
        if not AR_ORDER : ar_para = []
        else : ar_para = ARIMA_results.arparams
        
        del ARIMA_model, ARIMA_results, wls_res
        #=================================================================
        # KF init
        mu0 = np.append(wls_m[0: len(wls_m) - len(post_decy)].copy(), np.zeros([AR_ORDER]))
        mu0[0] = gps_y[0]
        
        time_dim = len(gps_full_decy)
        state_dim = len(mu0)
        
        cov0 = np.eye(state_dim) * 1e3
        cov0[-AR_ORDER+1, -AR_ORDER+1] = 0
        cov0[-AR_ORDER+2, -AR_ORDER+2] = 0
        cov0[0, 0] = 100
        cov0[2, 2] = cov0[3, 3] = cov0[4, 4] = cov0[5, 5]= 100
        
        dt = np.array([gps_full_decy[i+1] - gps_full_decy[i]
                      for i in range(time_dim-1)], dtype = float)
        
        # interpolate y_sig
        gps_full_y_sig = np.interp(gps_full_decy, gps_full_decy[~np.isnan(gps_full_y_sig)],     
                                   gps_full_y_sig[~np.isnan(gps_full_y_sig)])
        
        del post_decy, wls_m
        #==========================================
        # SSM init
        ar_start_ind = 6 + len(offset_decy)
        Q = get_Q(time_dim, state_dim)
        
        F = get_F(time_dim, state_dim, dt, ar_para, ar_start_ind, AR_ORDER)
        H = get_H(time_dim, state_dim, gps_full_decy, AR_ORDER, offset_decy)
        R = get_R(gps_full_y_sig)
        
        #==========================================
        # PSO
        mu = np.zeros((time_dim, state_dim, 1))
        mu[0,:,0] = mu0
        
        cov = np.zeros((time_dim, state_dim, state_dim))
        cov[0,:,:] = cov0
        
        '''
        pso_score, pso_theta = mle_apso(p_num, itr, theta_num, lb, ub, 
                             time_dim, state_dim, gps_full_y, gps_full_y_sig, 
                             mu, cov, H, R, F, Q, post_ind, AR_ORDER)
        
        if os.path.exists(OUTPUT_DIR + name + "/pso.txt"):
            pso = import_data(OUTPUT_DIR + name, "/pso.txt")
            pso_logL = float(pso[0][-1])
            
            del pso
            if pso_logL<pso_score[-1]:
                del pso_logL, cov, cov0, dt, F, freq, freq_cpy, freq_omega_cpy, \
                    gps_full_decy, gps_full_y, gps_full_y_sig, gps_y, gps_y_sig, H, \
                    mu, mu0, offset_decy, post_ind, pso_score, pso_theta, Q, R, state_dim
                    
                gc.collect()
                continue
        ''' 
    
        pso = import_data(OUTPUT_DIR + name, "/pso.txt")
        pso_logL = float(pso[0][-1])
        pso_theta = np.array(pso[-1], dtype=float)
        pso_theta[0] = 0
        pso_theta[-2] = 0
        
        
        
        #print("-2logL : " + str(pso_score[-1]))
        
        Q = get_Q(time_dim, state_dim)
        R = get_R(gps_full_y_sig)
        
        Q[:, 1, 1] = pso_theta[0]
        Q[:, 2, 2] = pso_theta[1]
        Q[:, 3, 3] = pso_theta[1]
        Q[:, 4, 4] = pso_theta[2]
        Q[:, 5, 5] = pso_theta[2]
        Q[:, -AR_ORDER, -AR_ORDER] = R[1::] * pso_theta[3]
        
        for p in range(len(post_ind)):
            Q[post_ind[p]-1, 1, 1] = 1e6
        
        R = R * pso_theta[4]
        white_amp = np.sqrt(np.mean(R))
        
        del post_ind
        #==========================================
        # KF
        mu = np.zeros((time_dim, state_dim, 1))
        mu[0,:,0] = mu0
        
        cov = np.zeros((time_dim, state_dim, state_dim))
        cov[0,:,:] = cov0
    
        mu, cov, logL = kf_main(time_dim, state_dim, gps_full_y, mu, cov, H, R, F, Q)   
        
        H_plot = H.copy()
        H_plot[:, 0, -AR_ORDER] = 0
        kf_y = [(a @ b)[0, 0] for a,b in zip(H_plot, mu)]
        
        # stablelize inv 
        for i in range(state_dim):
            Q[:, i, i] = Q[:, i, i] + EPS
            
        if len(offset_decy):
            offset_x_ind = [6+i for i in range(len(offset_decy))]
            Q[:, offset_x_ind, offset_x_ind] = 0
            
            del offset_x_ind
        
        # rts main
        mu, cov, rts_y = rts(time_dim, mu, cov, F, Q, H_plot, AR_ORDER)
        
        rts_sig_diag = np.sqrt([np.diag(a) for a in cov])
        #==============================================
        # Output format
        kf_trend = mu[:, 0, 0]
        
        if len(offset_decy):
            kf_offset = H[:, 0, 6:7+len(offset_decy)] @ mu[0, 6:7+len(offset_decy), 0]
            kf_trend = kf_trend + kf_offset
            
            del kf_offset
            
        kf_annual = [h[0, 2:4] @ m[2:4, 0] for h, m in zip(H, mu)]
        kf_semi_annual = [h[0, 4:6] @ m[4:6, 0] for h, m in zip(H, mu)]
        
        kf_color = mu[:, -AR_ORDER, 0]
        kf_res = gps_full_y - rts_y - mu[:, -3, 0]
        psd_lom_kf = lombscargle(gps_full_decy[~np.isnan(kf_res)],\
                              kf_res[~np.isnan(kf_res)], freq_omega_cpy)/365.25*2
        
        rts_y_var = np.array([np.sqrt((a @ b @ a.T)[0, 0]) for a, b in zip(H_plot, cov)])
        
        
        v_sig = np.sqrt(np.mean(cov[:, 1, 1]))
        v_sig_list.append(v_sig)
        #==============================================
        # Plot        
        del cov, cov0, dt, F, freq, freq_cpy, freq_omega_cpy, gps_full_decy, gps_full_y,\
            gps_full_y_sig, gps_y, gps_y_sig, H, H_plot, i, kf_annual, kf_color, kf_res,\
            kf_semi_annual, kf_trend, kf_y, logL, mu, mu0, offset_decy, psd_lom_kf, \
            pso_theta, Q, R, rts_sig_diag, rts_y, rts_y_var, state_dim, \
            time_dim, white_amp
            
        gc.collect()
   
import matplotlib.pyplot as plt
plt.plot(t_list, v_wls_sig_list, 'o', ms=2)  
plt.plot(t_list, v_sig_list, 'o', ms=2) 


import matplotlib.pyplot as plt
plt.loglog(t_list, v_wls_sig_list, 'o', ms=2)  
plt.loglog(t_list, v_sig_list, 'o', ms=2) 