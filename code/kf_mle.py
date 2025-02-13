# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:32:33 2024

@author: cdrg
"""

import os
import numpy as np
from numba import jit
from scipy.optimize import minimize
#import multiprocessing

#=====================================================================
# Dir
E_data_dir = "D:/RA_all/taiwan_all_ls_post/output/kf_ar3/E-W/"
N_data_dir = "D:/RA_all/taiwan_all_ls_post/output/kf_ar3/N-S/"
U_data_dir = "D:/RA_all/taiwan_all_ls_post/output/kf_ar3/U-D/"

data_list = os.listdir(E_data_dir)
noise_filname = "noise.txt"

output_dir = "D:/RA_all/taiwan_all_ls_post/output/kf_noise/"

#=====================================================================
# Function
def import_data(path, filename):
    data = []
    with open(path + filename) as file:
        for line in file:
            data.append(line.rstrip().split())
            
    return data
    del line, file


@jit(nopython=True)
def create_C(sigma_pl, kappa, dt):
    N = len(dt)
    U = np.identity(N)
    h = np.zeros(N)
    h[0] = 1
    
    for i in range(1, N):
        h[i] = (i - kappa / 2 - 1) / i * h[i-1]
        
        for j in range(0, N - i):
            U[j, j + i] = h[i]
    
    scale_factor = np.diag(dt ** (-kappa / 4))
    U = sigma_pl * U.T @ scale_factor
    
    return U @ U.T


@jit(nopython=True)
def log_likelihood(theta, noise, dt):
    N = len(noise)
    
    C = create_C(theta[0], theta[1], dt)
    r = noise.reshape((N, 1))
    
    U = np.linalg.cholesky(C).T
    ln_det_C = 0.0
    
    for i in range(0, N):
        ln_det_C += 2 * np.log(U[i, i])
    
    U_inv_r = np.linalg.solve(U, r)
    r_T_C_inv_r = np.dot(U_inv_r.T, U_inv_r)[0, 0]
    logL = -ln_det_C - r_T_C_inv_r
    
    return -logL


def run_minimization(args):
    initial_guess, color, dt = args
    res = minimize(
        log_likelihood, 
        initial_guess, 
        args=(color, dt), 
        method='Nelder-Mead', 
        bounds=[(0.001, None), (-3.5, 1.5)], 
        tol=0.01
    )
    return res

#=====================================================================
if __name__ == "__main__":
    for st in data_list:
        if st not in ["SUNM"]:continue
        
        E_data = import_data(E_data_dir + st + "/", noise_filname)
        N_data = import_data(N_data_dir + st + "/", noise_filname)
        U_data = import_data(U_data_dir + st + "/", noise_filname)
        
        E_white_amp = float(E_data[0][-1])
        N_white_amp = float(N_data[0][-1])
        U_white_amp = float(U_data[0][-1])
        
        E_data = np.array(E_data[2::], dtype=float)
        N_data = np.array(N_data[2::], dtype=float)
        U_data = np.array(U_data[2::], dtype=float)
        
        E_color, N_color, U_color = [data[:, 1] for data in (E_data, N_data, U_data)]
        
        # impute -> nan
        time = E_data[:, 0]
        nan_ind = np.isnan(E_data[:, 2])
        if len(time) < 365.25 * 2 : continue
        
        time = time[~nan_ind]
        if len(time) / len(nan_ind) < 0.5 : continue
        
        E_color = E_color[~nan_ind]
        N_color = N_color[~nan_ind]
        U_color = U_color[~nan_ind]
        
        dt = np.array([1/365.25]+[time[i+1]-time[i] for i in range(len(time)-1)],dtype=float)
        
        #===================================
        
        res_E = minimize(log_likelihood, np.array([10, -1.2]), args=(E_color, dt),\
                         method='Nelder-Mead', bounds=[(0.001, None),(-3.5, 1.5)], tol=0.01) 
        
        color_amp_E = res_E['x'][0]
        color_kappa_E = res_E['x'][1]
        
        #===================================
        res_N = minimize(log_likelihood, np.array([color_amp_E, color_kappa_E]), args=(N_color, dt),\
                         method='Nelder-Mead', bounds=[(0.001, None),(-3.5, 1.5)], tol=0.01) 
            
        color_amp_N = res_N['x'][0]
        color_kappa_N = res_N['x'][1]
        
        #===================================
        res_U = minimize(log_likelihood, np.array([10, -1]), args=(U_color, dt),\
                         method='Nelder-Mead', bounds=[(0.001, None),(-3.5, 1.5)], tol=0.01) 
            
        color_amp_U = res_U['x'][0]
        color_kappa_U = res_U['x'][1]
         
        #===================================
        # 3CPU
        '''
        tasks = [
            (np.array([8, -1.2]), E_color, dt),
            (np.array([8, -1.2]), N_color, dt),
            (np.array([10, -1.0]), U_color, dt)
        ]
        
        with multiprocessing.Pool(processes=3) as pool:
            results = pool.map(run_minimization, tasks)
        
        color_amp_E = results[0]['x'][0]
        color_kappa_E = results[0]['x'][1]
        
        color_amp_N = results[1]['x'][0]
        color_kappa_N = results[1]['x'][1]
        
        color_amp_U = results[2]['x'][0]
        color_kappa_U = results[2]['x'][1]
        '''
        #===================================
        # output
        with open(output_dir+st+".txt",'w') as f:
            f.write("{0:<25}    {1:<25}    {2:<25}\n"\
                    .format("color_k", "color_sig(mm/yr^(-k/4))", "white_sig(mm)"))
    
            f.write("{0:<25}    {1:<25}    {2:<25}\n"\
                    .format(str(color_kappa_E), str(color_amp_E), str(E_white_amp)))
                
            f.write("{0:<25}    {1:<25}    {2:<25}\n"\
                    .format(str(color_kappa_N), str(color_amp_N), str(N_white_amp)))
                
            f.write("{0:<25}    {1:<25}    {2:<25}"\
                    .format(str(color_kappa_U), str(color_amp_U), str(U_white_amp)))
    



