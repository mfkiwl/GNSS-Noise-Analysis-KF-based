# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:46:16 2024

@author: cdrg
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def kf_predict(mu, cov, F, Q):
    mu_new = F @ mu
    cov_new = F @ cov @ F.T + Q
    
    return mu_new, cov_new


@jit(nopython=True)
def kf_update(mu, cov, H, R, y, mu_dim):
    # Innovation
    v = y - H @ mu
    S = H @ cov @ H.T + R
    
    # Kalman gain
    #K = cov @ H.T @ np.linalg.inv(S)
    K = cov @ H.T / S[0, 0]
    
    # update
    new_mu = mu + K @ v
    new_cov = (np.eye(mu_dim) - K @ H) @ cov
    
    #logL = np.log(S) + v.T @ np.linalg.inv(S) @ v
    logL = np.log(S) + v.T @ v / S[0, 0]
    
    return new_mu, new_cov, logL
    

@jit(nopython=True)
def kf_main(time_dim, state_dim, y, mu, cov, H, R, F, Q):
    joint_logL = 0
    
    for t in range(time_dim):
        # if y[t]==y[t]:
        if not np.isnan(y[t]):
            mu[t], cov[t], logL = kf_update(mu[t], cov[t], H[t],
                                      R[t], y[t], state_dim)
            
            joint_logL += logL[0, 0]
            
        if t<time_dim-1:
            mu[t+1], cov[t+1] = kf_predict(mu[t], cov[t], F[t], Q[t])
                      
    return mu, cov, joint_logL


@jit(nopython=True)
def rts(time_dim, mu, cov, F, Q, H, ar_order):
    rts_y_mean = np.zeros(time_dim)
    
    for i in range(time_dim-1, -1, -1):
        if i < time_dim - 1:
            P_pred = F[i] @ cov[i] @ F[i].T + Q[i]
            K = cov[i] @ F[i].T @ np.linalg.inv(P_pred)
            mu[i] += K @ (mu[i+1] - (F[i] @ mu[i]))
            cov[i] += K @ (cov[i+1] - P_pred) @ K.T
            
        rts_y_mean[i] = (H[i] @ mu[i])[0, 0]
        
    return mu, cov, rts_y_mean

'''
@jit(nopython=True)
def mle_apso(p_num, itr, theta_num, lb, ub, 
             time_dim, state_dim, y, y_sig, mu, cov, H, R, F, Q):
    
    max_v = 0.2 * ub
    
    # partical position
    x = np.zeros((p_num, theta_num))   
    for i in range(theta_num):
        x[:,i] = np.random.uniform(lb[i], ub[i], p_num)
    
    # particle velocity
    v = -max_v + 2 * max_v * np.random.rand(p_num, theta_num)
    
    # particle score
    f = np.zeros(p_num)    
    pbest_f = np.zeros(p_num)
    pbest_x = np.zeros((p_num, theta_num))
    
    gbest_f = 0
    gbest_x = np.zeros(theta_num)
    
    loss_curve = np.zeros(itr)
    
    for i in range(itr):
        for j in range(p_num):
            for t in range(time_dim):
                Q[t, 1, 1] = x[j, 0]
                Q[t, 2, 2] = x[j, 1]
                Q[t, 3, 3] = x[j, 1]
                Q[t, 4, 4] = x[j, 2]
                Q[t, 5, 5] = x[j, 2]                
                Q[t, 6, 6] = y_sig[t]**2 * x[j, 3]
                
                R[t] = y_sig[t]**2 * x[j, 4]
             
            _, _, f[j] = kf_main(time_dim, state_dim, y, mu, cov, H, R, F, Q)
            
            if f[j] < pbest_f[j] or i==0:
                pbest_f[j] = f[j]
                pbest_x[j] = x[j]
                
        if np.min(f) < gbest_f or i==0:
            idx = f.argmin()
            gbest_x = x[idx]
            gbest_f = f.min()
            
        loss_curve[i] = gbest_f 
   
    return gbest_x, gbest_f, loss_curve
    
'''    
    

@jit(nopython=True)
def mle_apso(p_num, itr, theta_num, lb, ub, 
             time_dim, state_dim, y, y_sig, mu, cov, H, R, F, Q, post_ind, ar_order):
    
    w_max = 0.9
    w_min = 0.3
    w_diff = w_max - w_min
    
    c1 = c2 = 2
    max_v = 0.2 * ub
    
    #==========================================================
    # particle
    x = np.zeros((p_num, theta_num))   
    for i in range(theta_num):
        x[:,i] = np.random.uniform(lb[i], ub[i], p_num)

    #==========================================================
    # velocity
    v = -max_v + 2*max_v*np.random.rand(p_num, theta_num)
    
    #==========================================================
    # init
    f = np.zeros(p_num)
    f_new = np.zeros(p_num)
    f_best = np.zeros(itr)
       
    post_num = post_ind.shape[0]
    
    for n in range(p_num):
        for t in range(time_dim-1):
            Q[t, 1, 1] = x[n, 0]
            Q[t, 2, 2] = x[n, 1]
            Q[t, 3, 3] = x[n, 1]
            Q[t, 4, 4] = x[n, 2]
            Q[t, 5, 5] = x[n, 2]
            Q[t, -ar_order, -ar_order] = y_sig[t+1]**2 * x[n, 3]
            
            R[t] = y_sig[t]**2 * x[n, 4]
                      
        if post_num:
            for p in range(post_num):                
                Q[int(post_ind[p]-1), 1, 1] = 1e6
        
        _, _, f[n] = kf_main(time_dim, state_dim, y, mu, cov, H, R, F, Q)
    
    pbest = x.copy()
    ind = np.argmin(f)
    gbest_value = min(f)
    gbest = x[ind,:]
    
    #==========================================================
    # PSO
    for i in range(itr):
        w = w_max-(w_diff*i)/(itr-1)
        for n in range(p_num):
            for j in range(theta_num):
                # update velocity
                v[n, j] = w * v[n, j] + \
                          c1 * np.random.rand() * (pbest[n, j] - x[n, j]) + \
                          c2 * np.random.rand() * (gbest[j] - x[n, j])
                         
                # clip velocity
                if v[n,j] < -max_v[j] : v[n,j] = -max_v[j]            
                elif v[n,j] > max_v[j] : v[n,j] = max_v[j]
                
                #===============================================
                # update particle
                x[n, j] = x[n, j] + v[n, j]
                
                # clip particle
                if x[n,j] < lb[j] : x[n,j] = lb[j]
                elif x[n,j] > ub[j] : x[n,j] = ub[j]
                
            #==================================================
            for t in range(time_dim-1):
                Q[t, 1, 1] = x[n, 0]
                Q[t, 2, 2] = x[n, 1]
                Q[t, 3, 3] = x[n, 1]
                Q[t, 4, 4] = x[n, 2]
                Q[t, 5, 5] = x[n, 2]
                Q[t, -ar_order, -ar_order] = y_sig[t+1]**2 * x[n, 3]
                
                R[t] = y_sig[t]**2 * x[n, 4]
                
            if post_num:
                for p in range(post_num):                
                    Q[int(post_ind[p]-1), 1, 1] = 1e6
                    
            _, _, f_new[n] = kf_main(time_dim, state_dim, y, mu, cov, H, R, F, Q)
            
            if f_new[n] < f[n]:
                pbest[n] = x[n]
                f[n] = f_new[n]
        
        if np.min(f) < gbest_value:
            ind = f.argmin()
            gbest = pbest[ind]
            gbest_value = f.min()
            
        f_best[i] = gbest_value
        
    return f_best, gbest
    
    
    

    
    
    
    
    
    
    
    