# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:22:12 2024

@author: cdrg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

#=========================================================
#init
kf_path = "D:/RA_all/taiwan_all_ls_post/output/kf_noise/"
kf_list = [a for a in os.listdir(kf_path) if a[-3::] == 'txt']

data_num = len(kf_list)

hector_path_E = "D:/RA_all/taiwan_all_ls_post/output/hector/output/E/"
hector_path_N = "D:/RA_all/taiwan_all_ls_post/output/hector/output/N/"
hector_path_U = "D:/RA_all/taiwan_all_ls_post/output/hector/output/U/"

hector_list = [a for a in os.listdir(kf_path) if a[-3::] == 'txt']
outupt_path = "D:/RA_all/taiwan_all_ls_post/output/pic/"
#=========================================================
# func
def import_data(noise_path, noise_list):
    data = []
    with open(noise_path + noise_list) as file:
        for line in file:
            data.append(line.rstrip().split())

    return data

#=========================================================
# import data
kf_E_list = np.full([data_num, 3], np.nan)
kf_N_list = np.full([data_num, 3], np.nan)
kf_U_list = np.full([data_num, 3], np.nan)

hector_E_list = np.full([data_num, 3], np.nan)
hector_N_list = np.full([data_num, 3], np.nan)
hector_U_list = np.full([data_num, 3], np.nan)

d_p = []
for i, st in enumerate(kf_list):    
    kf_data = [a for a in import_data(kf_path, st) if len(a)!=0]

    kf_k_E = float(kf_data[1][0])
    kf_color_amp_E = float(kf_data[1][1])
    kf_white_amp_E = float(kf_data[1][2])
    
    kf_k_N = float(kf_data[2][0])
    kf_color_amp_N = float(kf_data[2][1])
    kf_white_amp_N = float(kf_data[2][2])
    
    kf_k_U = float(kf_data[3][0])
    kf_color_amp_U = float(kf_data[3][1])
    kf_white_amp_U = float(kf_data[3][2])
    
    kf_E_list[i][0] = kf_k_E
    kf_E_list[i][1] = kf_color_amp_E
    kf_E_list[i][2] = kf_white_amp_E
    
    kf_N_list[i][0] = kf_k_N
    kf_N_list[i][1] = kf_color_amp_N
    kf_N_list[i][2] = kf_white_amp_N
    
    kf_U_list[i][0] = kf_k_U
    kf_U_list[i][1] = kf_color_amp_U
    kf_U_list[i][2] = kf_white_amp_U
    
    hector_data_E = [a for a in import_data(hector_path_E, st) if len(a)!=0]
    hector_data_N = [a for a in import_data(hector_path_N, st) if len(a)!=0]
    hector_data_U = [a for a in import_data(hector_path_U, st) if len(a)!=0]
    
    data_num_ponits = float([a for a in hector_data_E if a[0]=="Number"][0][-1])
    d_p.append(data_num_ponits)
    
    if st in ["CHIN.txt", "DANI.txt", "DASI.txt", "DSIN.txt", "ERLN.txt", 
              "FENP.txt", "FKDO.txt", "GAIS.txt", "GS10.txt", "GS22.txt",
              "GS28.txt", "GS43.txt", "GS47.txt", "GS66.txt", "GS68.txt", 
              "GS69.txt", "GS74.txt", "GS94.txt", "GS95.txt", "HCHM.txt",
              "HGC1.txt", "HOBE.txt", "HOPN.txt", "KSH2.txt", "LIKN.txt", 
              "LIUC.txt", "LIYU.txt", "MITO.txt", "MUDA.txt", "NDH9.txt",
              "PUSN.txt", "SAME.txt", "SANI.txt", "SEED.txt", "SHLU.txt",
              "SNES.txt", "SUAN.txt", "TASI.txt", "TATA.txt", "TASO.txt", 
              "W021.txt", "W029.txt", "WIAN.txt", "YILN.txt"] : continue
    
    hector_k_E = [a for a in hector_data_E if a[0]=="kappa"]
    hector_amp_E = [a for a in hector_data_E if a[0]=="sigma"]
    
    hector_k_N = [a for a in hector_data_N if a[0]=="kappa"]
    hector_amp_N = [a for a in hector_data_N if a[0]=="sigma"]
    
    hector_k_U = [a for a in hector_data_U if a[0]=="kappa"]
    hector_amp_U = [a for a in hector_data_U if a[0]=="sigma"]
    
    if len(hector_k_E)!=0:
        hector_k_E = float(hector_k_E[0][2])
        hector_amp_E = [float(a[2]) for a in hector_amp_E]
        
        color_amp_E = hector_amp_E[0]
        wh_amp_E = hector_amp_E[1]
        
        hector_E_list[i][0] = hector_k_E
        hector_E_list[i][1] = color_amp_E
        hector_E_list[i][2] = wh_amp_E
        
    else:
        hector_E_list[i][0] = np.nan
        hector_E_list[i][1] = np.nan
        hector_E_list[i][2] = np.nan
        
    if len(hector_k_N)!=0:
        hector_k_N = float(hector_k_N[0][2])
        hector_amp_N = [float(a[2]) for a in hector_amp_N]
        
        color_amp_N = hector_amp_N[0]
        wh_amp_N = hector_amp_N[1]
        
        hector_N_list[i][0] = hector_k_N
        hector_N_list[i][1] = color_amp_N
        hector_N_list[i][2] = wh_amp_N
        
    else:
        hector_N_list[i][0] = np.nan
        hector_N_list[i][1] = np.nan
        hector_N_list[i][2] = np.nan
        
    if len(hector_k_U)!=0:
        hector_k_U = float(hector_k_U[0][2])
        hector_amp_U = [float(a[2]) for a in hector_amp_U]
        
        color_amp_U = hector_amp_U[0]
        wh_amp_U = hector_amp_U[1]
        
        hector_U_list[i][0] = hector_k_U
        hector_U_list[i][1] = color_amp_U
        hector_U_list[i][2] = wh_amp_U
        
    else:
        hector_U_list[i][0] = np.nan
        hector_U_list[i][1] = np.nan
        hector_U_list[i][2] = np.nan
        
#=========================================================
# output
d_p = np.array(d_p, dtype=float)

fig = plt.figure(figsize=(36, 12))
ax0 = plt.subplot2grid((1, 3), (0, 0))
ax1 = plt.subplot2grid((1, 3), (0, 1))
ax2 = plt.subplot2grid((1, 3), (0, 2))
axes = [ax0, ax1, ax2]

aaa = hector_U_list.copy()
bbb = kf_U_list.copy()

ax0.plot(aaa[d_p>365.25*2, 0], bbb[d_p>365.25*2, 0], 'o', c=[0.2, 0.2, 0.8], ms=7, label='<5 yr')
ax0.plot(aaa[d_p>365.25*5, 0], bbb[d_p>365.25*5, 0], 'o', c=[0.5, 0.7, 0.2], ms=7, label='<10 yr')
ax0.plot(aaa[d_p>365.25*10, 0], bbb[d_p>365.25*10, 0], 'o', c=[0.9, 0.9, 0.05], ms=7, label='<15 yr')
ax0.plot(aaa[d_p>365.25*15, 0], bbb[d_p>365.25*15, 0], 'o', c=[0.9, 0.35, 0.10], ms=7, label='<20 yr')
ax0.plot(aaa[d_p>365.25*20, 0], bbb[d_p>365.25*20, 0], 'o', c=[0.8, 0.15, 0.10], ms=7, label='>20 yr')
#ax0.plot(aaa[:, 0], bbb[:, 0], 'o', c='k', ms=6.6)
ax0.plot(np.arange(-4, 1, 0.1), np.arange(-4, 1, 0.1), ls='--',color='k', lw=3.5)
ax0.yaxis.set_minor_locator(AutoMinorLocator(5))
ax0.xaxis.set_minor_locator(AutoMinorLocator(5))

ax1.plot(aaa[d_p>365.25*2, 1], bbb[d_p>365.25*2, 1], 'o', c=[0.2, 0.2, 0.8], ms=5, label='<5 yr')
ax1.plot(aaa[d_p>365.25*5, 1], bbb[d_p>365.25*5, 1], 'o', c=[0.5, 0.7, 0.2], ms=5, label='<10 yr')
ax1.plot(aaa[d_p>365.25*10, 1], bbb[d_p>365.25*10, 1], 'o', c=[0.9, 0.9, 0.05], ms=5, label='<15 yr')
ax1.plot(aaa[d_p>365.25*15, 1], bbb[d_p>365.25*15, 1], 'o', c=[0.9, 0.35, 0.10], ms=5, label='<20 yr')
ax1.plot(aaa[d_p>365.25*20, 1], bbb[d_p>365.25*20, 1], 'o', c=[0.8, 0.15, 0.10], ms=5, label='>20 yr')
#ax1.plot(aaa[:, 1], bbb[:, 1], 'o', c='k', ms=4.7)
ax1.plot(np.arange(-1, 61, 1), np.arange(-1, 61, 1), ls='--',color='k', lw=3.5)
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

ax2.plot(aaa[d_p>365.25*2, 2], bbb[d_p>365.25*2, 2], 'o', c=[0.2, 0.2, 0.8], ms=5.5, label='<5 yr')
ax2.plot(aaa[d_p>365.25*5, 2], bbb[d_p>365.25*5, 2], 'o', c=[0.5, 0.7, 0.2], ms=5.5, label='<10 yr')
ax2.plot(aaa[d_p>365.25*10, 2], bbb[d_p>365.25*10, 2], 'o', c=[0.9, 0.9, 0.05], ms=5.5, label='<15 yr')
ax2.plot(aaa[d_p>365.25*15, 2], bbb[d_p>365.25*15, 2], 'o', c=[0.9, 0.35, 0.10], ms=5.5, label='<20 yr')
ax2.plot(aaa[d_p>365.25*20, 2], bbb[d_p>365.25*20, 2], 'o', c=[0.8, 0.15, 0.10], ms=5.5, label='>20 yr')
#ax2.plot(aaa[:, 2], bbb[:, 2], 'o', c='k', ms=4.7)
ax2.plot(np.arange(-1, 31, 1), np.arange(-1, 31, 1), ls='--',color='k', lw=3.5)
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

[ax.tick_params(axis='both', which='major', direction='in',
                bottom=True, top=True, right=True, left=True,
                length=15, width=2.8, labelsize=22, pad=11) for ax in axes]

[ax.tick_params(axis='both', which='minor', direction='in',
                bottom=True, top=True, right=True, left=True,
                length=9, width=1.9, labelsize=22, pad=11) for ax in axes]

[ax.spines[b].set_linewidth(3.7) for b in ['top', 'bottom', 'left', 'right']for ax in axes]
[ax.spines[b].set_color("black") for b in ['top', 'bottom', 'left', 'right']for ax in axes]

ax0.set_title("U-D Component Spectral Index", fontsize=36, pad=28)
ax0.set_xlabel("Hector spectral index ($\kappa$)", fontsize=27, labelpad=14)
ax0.set_ylabel("Kalman spectral index ($\kappa$)", fontsize=27, labelpad=14)

ax1.set_title("U-D Color Noise Amplitude", fontsize=36, pad=28)
ax1.set_xlabel("Hector color noise amplitude (mm/$yr^{-\kappa/4}$)", fontsize=27, labelpad=14)
ax1.set_ylabel("Kalman color noise amplitude (mm/$yr^{-\kappa/4}$)", fontsize=27, labelpad=14)

ax2.set_title("U-D White Noise Amplitude", fontsize=36, pad=28)
ax2.set_xlabel("Hector white noise amplitude (mm)", fontsize=27, labelpad=14)
ax2.set_ylabel("Kalman white noise amplitude (mm)", fontsize=27, labelpad=14)

ax0.set_xlim(-3.1, 0.1)
ax0.set_ylim(-3.1, 0.1)

ax1.set_xlim(-0.5, 40)
ax1.set_ylim(-0.5, 40)

ax2.set_xlim(-0.5, 15)
ax2.set_ylim(-0.5, 15)

ax1.legend(fontsize=25)

ax0.set_aspect('equal')
ax1.set_aspect('equal')
ax2.set_aspect('equal')
plt.tight_layout(pad=3.5)
plt.savefig(outupt_path + "kf_hector_U_color.png", dpi=400)