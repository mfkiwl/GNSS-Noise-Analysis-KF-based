# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:30:02 2024

@author: cdrg
"""

from random import gauss
import numpy as np
import matplotlib.pyplot as plt

#=====================================================
DATA_POINTS = 50000
DT = 1/365.25

start_time = 2010
time = np.array([start_time + i * DT for i in range(DATA_POINTS)])

#=====================================================
# Trend
def irw(x0, v0, acc_sig, data_points, dt):
    acc = np.zeros(data_points)
    vel = np.zeros(data_points)
    x = np.zeros(data_points)
    
    acc[0] = gauss(0, acc_sig)
    vel[0] = v0
    x[0] = x0
    
    for i in range(1, data_points):
        acc[i] = gauss(0, acc_sig)
        vel[i] = vel[i - 1] + acc[i]
        x[i] = x[i - 1] + vel[i] * dt
        
    return x


x0 = 0
v0 = 0
acc_sig = 0.8
trend = irw(x0, v0, acc_sig, DATA_POINTS, DT)


def white_noise(sig, data_points):
    return np.array([gauss(0, sig) for _ in range(DATA_POINTS)])


noise_sig = 0.9
noise = white_noise(noise_sig, DATA_POINTS)
noise_sig_arr = np.full(DATA_POINTS, noise_sig)

data = trend + noise

#=====================================================
# KF
def get_Q(t_dim, x_dim):
    return np.zeros((t_dim-1, x_dim, x_dim))


def get_F(t_dim, x_dim, dt):
    F = np.tile(np.eye(x_dim), (t_dim-1, 1, 1))
    F[:, 0, 1] = dt
    
    return F


def get_H(t_dim, x_dim, time):
    H = np.zeros((t_dim, 1, x_dim))
    
    for t in range(len(time)):
        H[t, 0, 0] = 1
       
    return H


def get_R(sig):
    return sig**2


# Kalman
def kf_predict(mu, cov, F, Q):
    mu_new = F @ mu
    cov_new = F @ cov @ F.T + Q
    
    return mu_new, cov_new


def kf_update(mu, cov, H, R, y, mu_dim):
    # Innovation
    v = y - H @ mu
    S = H @ cov @ H.T + R
    
    # Kalman gain
    K = cov @ H.T / S[0, 0]
    
    # update
    new_mu = mu + K @ v
    new_cov = (np.eye(mu_dim) - K @ H) @ cov
    
    #logL = np.log(S) + v.T @ np.linalg.inv(S) @ v
    logL = np.log(S) + v.T @ v / S[0, 0]
    
    return new_mu, new_cov, logL


def kf_main(time_dim, state_dim, y, mu, cov, H, R, F, Q):
    joint_logL = 0
    
    for t in range(time_dim):
        if not np.isnan(y[t]):
            mu[t], cov[t], logL = kf_update(mu[t], cov[t], H[t],
                                      R[t], y[t], state_dim)
            
            joint_logL += logL[0, 0]
            
        if t<time_dim-1:
            mu[t+1], cov[t+1] = kf_predict(mu[t], cov[t], F[t], Q[t])
                      
    return mu, cov, joint_logL


def rts(time_dim, mu, cov, F, Q, H):
    rts_y_mean = np.zeros(time_dim)
    
    for i in range(time_dim-1, -1, -1):
        if i < time_dim - 1:
            P_pred = F[i] @ cov[i] @ F[i].T + Q[i]
            K = cov[i] @ F[i].T @ np.linalg.inv(P_pred)
            mu[i] += K @ (mu[i+1] - (F[i] @ mu[i]))
            cov[i] += K @ (cov[i+1] - P_pred) @ K.T
            
        rts_y_mean[i] = (H[i] @ mu[i])[0, 0]
        
    return mu, cov, rts_y_mean


x_dim = 2

mu0 = np.array([[0], [0]])
cov0 = np.eye(2) * 10

mu = np.zeros([DATA_POINTS, 2, 1])
cov = np.zeros([DATA_POINTS, 2, 2])

mu[0] = mu0
cov0 = cov0


F = get_F(DATA_POINTS, x_dim, DT)
H = get_H(DATA_POINTS, x_dim, time)

Q = get_Q(DATA_POINTS, x_dim)
Q[:, 1, 1] = acc_sig**2
Q[:, 0, 0] = 1e-12

R = get_R(noise_sig_arr)



kf_mu, kf_cov, joint_logL = kf_main(DATA_POINTS, x_dim, data, mu, cov, H, R, F, Q)
kf_y = np.array([(h @ x)[0, 0] for h, x in zip(H, mu)])

aa = np.array([(h @ x @ h.T)[0, 0] for h, x in zip(H, cov)])
aaa = np.sqrt(np.mean(R-aa))
print(np.std(data-kf_y))




rts_mu, rts_cov, rts_y_mean = rts(DATA_POINTS, mu, cov, F, Q, H)
print(np.std(data-rts_y_mean))

Q = np.append(Q, Q[:1], axis=0)

bb = np.array([(h @ x @ h.T)[0, 0] for h, x in zip(H, rts_cov)])
cc = np.array([(h @ x @ h.T)[0, 0] for h, x in zip(H, Q)])

dd = np.sqrt(np.mean(R + bb * cc / (aa + cc) - bb * aa / (aa + cc)))

'''
plt.plot(time, data, 'o', c='k', ms=1)
plt.plot(time, kf_y, c='r', lw=1)
'''


