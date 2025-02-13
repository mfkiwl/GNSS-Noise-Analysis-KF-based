# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:34:32 2024

@author: cdrg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

#=========================================================
#init
hector_path = "D:/RA_all/taiwan_all_ls_post/output/kf_noise/"
hector_list = [a for a in os.listdir(hector_path) if a[-3::] == 'txt']

data_num = len(hector_list)

outupt_path = "D:/RA_all/taiwan_all_ls_post/output/"
#=========================================================
def import_data(noise_path, noise_list):
    data = []
    with open(noise_path + noise_list) as file:
        for line in file:
            data.append(line.rstrip().split())

    return data

#=========================================================
# hector_E_list = np.zeros([data_num, 3])
# hector_N_list = np.zeros([data_num, 3])
# hector_U_list = np.zeros([data_num, 3])

hector_E_list = np.full([data_num, 3], np.nan)
hector_N_list = np.full([data_num, 3], np.nan)
hector_U_list = np.full([data_num, 3], np.nan)

for i, st in enumerate(hector_list):
    if st in ["CHIN.txt", "DANI.txt", "DASI.txt", "DSIN.txt", "ERLN.txt", 
              "FENP.txt", "FKDO.txt", "GAIS.txt", "GS10.txt", "GS22.txt",
              "GS28.txt", "GS43.txt", "GS47.txt", "GS66.txt", "GS68.txt", 
              "GS69.txt", "GS74.txt", "GS94.txt", "GS95.txt", "HCHM.txt",
              "HGC1.txt", "HOBE.txt", "HOPN.txt", "KSH2.txt", "LIKN.txt", 
              "LIUC.txt", "LIYU.txt", "MITO.txt", "MUDA.txt", "NDH9.txt",
              "PUSN.txt", "SAME.txt", "SANI.txt", "SEED.txt", "SHLU.txt",
              "SNES.txt", "SUAN.txt", "TASI.txt", "TATA.txt", "TASO.txt", 
              "W021.txt", "W029.txt", "WIAN.txt", "YILN.txt"] : continue
    
    hector_data = [a for a in import_data(hector_path, st) if len(a)!=0]

    hector_k_E = float(hector_data[1][0])
    hector_color_amp_E = float(hector_data[1][1])
    hector_white_amp_E = float(hector_data[1][2])
    
    hector_k_N = float(hector_data[2][0])
    hector_color_amp_N = float(hector_data[2][1])
    hector_white_amp_N = float(hector_data[2][2])
    
    hector_k_U = float(hector_data[3][0])
    hector_color_amp_U = float(hector_data[3][1])
    hector_white_amp_U = float(hector_data[3][2])
    
    hector_E_list[i][0] = hector_k_E
    hector_E_list[i][1] = hector_color_amp_E
    hector_E_list[i][2] = hector_white_amp_E
    
    hector_N_list[i][0] = hector_k_N
    hector_N_list[i][1] = hector_color_amp_N
    hector_N_list[i][2] = hector_white_amp_N
    
    hector_U_list[i][0] = hector_k_U
    hector_U_list[i][1] = hector_color_amp_U
    hector_U_list[i][2] = hector_white_amp_U
    
#=========================================================
fig = plt.figure(figsize=(25, 19))
ax0 = plt.subplot2grid((3, 3), (0, 0))
ax1 = plt.subplot2grid((3, 3), (0, 1))
ax2 = plt.subplot2grid((3, 3), (0, 2))

ax3 = plt.subplot2grid((3, 3), (1, 0))
ax4 = plt.subplot2grid((3, 3), (1, 1))
ax5 = plt.subplot2grid((3, 3), (1, 2))

ax6 = plt.subplot2grid((3, 3), (2, 0))
ax7 = plt.subplot2grid((3, 3), (2, 1))
ax8 = plt.subplot2grid((3, 3), (2, 2))
axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6 ,ax7, ax8]

hector_E_mean = np.mean(hector_E_list[~np.isnan(hector_E_list[:, 0])], axis=0)
hector_N_mean = np.mean(hector_N_list[~np.isnan(hector_E_list[:, 0])], axis=0)
hector_U_mean = np.mean(hector_U_list[~np.isnan(hector_E_list[:, 0])], axis=0)

ax0.hist(hector_E_list[:, 0], bins=np.arange(-3, 0.01, 0.1), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax0.yaxis.set_minor_locator(AutoMinorLocator(5))
ax0.xaxis.set_major_locator(MultipleLocator(0.5))
ax0.xaxis.set_minor_locator(MultipleLocator(0.1))

ax1.hist(hector_E_list[:, 1], bins=np.arange(0, 40, 1), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

ax2.hist(hector_E_list[:, 2], bins=np.arange(0, 15, 0.5), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

for ax, a in zip([ax0, ax1, ax2], hector_E_mean):
    ax.axvline(a, ls='-.', c='k', label="mean : "+str(round(a, 2)))
    
    
ax3.hist(hector_N_list[:, 0], bins=np.arange(-3, 0.01, 0.1), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
ax3.xaxis.set_major_locator(MultipleLocator(0.5))
ax3.xaxis.set_minor_locator(MultipleLocator(0.1))

ax4.hist(hector_N_list[:, 1], bins=np.arange(0, 40, 1), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax4.yaxis.set_minor_locator(AutoMinorLocator(5))
ax4.xaxis.set_minor_locator(AutoMinorLocator(5))

ax5.hist(hector_N_list[:, 2], bins=np.arange(0, 15, 0.5), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax5.yaxis.set_minor_locator(AutoMinorLocator(5))
ax5.xaxis.set_minor_locator(AutoMinorLocator(5))

for ax, a in zip([ax3, ax4, ax5], hector_N_mean):
    ax.axvline(a, ls='-.', c='k', label="mean : "+str(round(a, 2)))
    
ax6.hist(hector_U_list[:, 0], bins=np.arange(-3, 0.01, 0.1), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax6.yaxis.set_minor_locator(AutoMinorLocator(5))
ax6.xaxis.set_major_locator(MultipleLocator(0.5))
ax6.xaxis.set_minor_locator(MultipleLocator(0.1))

ax7.hist(hector_U_list[:, 1], bins=np.arange(0, 40, 1), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax7.yaxis.set_minor_locator(AutoMinorLocator(5))
ax7.xaxis.set_minor_locator(AutoMinorLocator(5))

ax8.hist(hector_U_list[:, 2], bins=np.arange(0, 15, 0.5), rwidth=0.85, color=[0.2, 0.2, 0.2])
ax8.yaxis.set_minor_locator(AutoMinorLocator(5))
ax8.xaxis.set_minor_locator(AutoMinorLocator(5))

for ax, a in zip([ax6, ax7, ax8], hector_U_mean):
    ax.axvline(a, ls='-.', c='k', lw=2, label="mean : "+str(round(a, 2)))
    
[ax.legend(fontsize=22) for ax in axes]

[ax.tick_params(axis='both', which='major', direction='in',
                bottom=True, top=True, right=True, left=True,
                length=12, width=2.4, labelsize=18, pad=10) for ax in axes]

[ax.tick_params(axis='both', which='minor', direction='in',
                bottom=True, top=True, right=True, left=True,
                length=6, width=1.8, labelsize=18, pad=10) for ax in axes]

[ax.spines[b].set_linewidth(2.4) for b in ['top', 'bottom', 'left', 'right'] for ax in axes]
[ax.spines[b].set_color("black") for b in ['top', 'bottom', 'left', 'right'] for ax in axes]

plt.suptitle("Kalman Noise Analysis", fontsize=40, y=0.98)
ax0.set_title("Spectral Index", fontsize=30, pad=15)
ax1.set_title("Color Noise Amplitude", fontsize=30, pad=15)
ax2.set_title("White Noise Amplitude", fontsize=30, pad=15)
'''
ax3.set_title("Spectral Index", fontsize=30, pad=15)
ax4.set_title("Hector N-S Component\nColor Noise Amplitude", fontsize=30, pad=15)
ax5.set_title("White Noise Amplitude", fontsize=30, pad=15)
'''
ax6.set_xlabel("spectral index ($\kappa$)", fontsize=23, labelpad=12)
ax7.set_xlabel("amplitude (mm/$yr^{-\kappa/4}$)", fontsize=23, labelpad=12)
ax8.set_xlabel("amplitude (mm)", fontsize=23, labelpad=13)
'''
ax3.set_xlabel("spectral index ($\kappa$)", fontsize=23, labelpad=12)
ax4.set_xlabel("amplitude (mm/$yr^{-\kappa/4}$)", fontsize=23, labelpad=12)
ax5.set_xlabel("amplitude (mm)", fontsize=23, labelpad=13)
'''
ax0.set_ylabel("E-W Component\nstation number", fontsize=25, labelpad=12)
ax3.set_ylabel("N-S Component\nstation number", fontsize=25, labelpad=12)
ax6.set_ylabel("U-D Component\nstation number", fontsize=25, labelpad=12)

ax0.set_xlim(-3, 0)
ax3.set_xlim(-3, 0)

ax0.set_ylim(0, 100)
ax3.set_ylim(0, 100)
ax6.set_ylim(0, 100)

ax1.set_ylim(0, 300)
ax4.set_ylim(0, 300)
ax7.set_ylim(0, 300)

ax2.set_ylim(0, 350)
ax5.set_ylim(0, 350)
ax8.set_ylim(0, 350)

plt.tight_layout(pad=3.5)
plt.savefig(outupt_path + "kf_hist.png", dpi=400)



fig = plt.figure(figsize=(36, 13))
ax0 = plt.subplot2grid((1, 3), (0, 0))
ax1 = plt.subplot2grid((1, 3), (0, 1))
ax2 = plt.subplot2grid((1, 3), (0, 2))
axes = [ax0, ax1, ax2]

d_p = np.array(d_p)
aaa = hector_E_list.copy()
bbb = hector_N_list.copy()

ax0.plot(aaa[d_p>365.25*2, 0], bbb[d_p>365.25*2, 0], 'o', c=[0.2, 0.2, 0.8], ms=7, label='<5 yr')
ax0.plot(aaa[d_p>365.25*5, 0], bbb[d_p>365.25*5, 0], 'o', c=[0.5, 0.7, 0.2], ms=7, label='<10 yr')
ax0.plot(aaa[d_p>365.25*10, 0], bbb[d_p>365.25*10, 0], 'o', c=[0.9, 0.9, 0.05], ms=7, label='<15 yr')
ax0.plot(aaa[d_p>365.25*15, 0], bbb[d_p>365.25*15, 0], 'o', c=[0.9, 0.35, 0.10], ms=7, label='<20 yr')
ax0.plot(aaa[d_p>365.25*20, 0], bbb[d_p>365.25*20, 0], 'o', c=[0.8, 0.15, 0.10], ms=7, label='>20 yr')

#ax0.plot(hector_E_list[:, 0], hector_N_list[:, 0], 'o', c='k', ms=6.6)
ax0.plot(np.arange(-4, 1, 0.1), np.arange(-4, 1, 0.1), ls='--',color='k', lw=3.5)
ax0.yaxis.set_minor_locator(AutoMinorLocator(5))
ax0.xaxis.set_minor_locator(AutoMinorLocator(5))

ax1.plot(aaa[d_p>365.25*2, 1], bbb[d_p>365.25*2, 1], 'o', c=[0.2, 0.2, 0.8], ms=5, label='<5 yr')
ax1.plot(aaa[d_p>365.25*5, 1], bbb[d_p>365.25*5, 1], 'o', c=[0.5, 0.7, 0.2], ms=5, label='<10 yr')
ax1.plot(aaa[d_p>365.25*10, 1], bbb[d_p>365.25*10, 1], 'o', c=[0.9, 0.9, 0.05], ms=5, label='<15 yr')
ax1.plot(aaa[d_p>365.25*15, 1], bbb[d_p>365.25*15, 1], 'o', c=[0.9, 0.35, 0.10], ms=5, label='<20 yr')
ax1.plot(aaa[d_p>365.25*20, 1], bbb[d_p>365.25*20, 1], 'o', c=[0.8, 0.15, 0.10], ms=5, label='>20 yr')
#ax1.plot(hector_E_list[:, 1], hector_N_list[:, 1], 'o', c='k', ms=4.7)
ax1.plot(np.arange(-1, 51, 1), np.arange(-1, 51, 1), ls='--',color='k', lw=3.5)
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

ax2.plot(aaa[d_p>365.25*2, 2], bbb[d_p>365.25*2, 2], 'o', c=[0.2, 0.2, 0.8], ms=3.5, label='<5 yr')
ax2.plot(aaa[d_p>365.25*5, 2], bbb[d_p>365.25*5, 2], 'o', c=[0.5, 0.7, 0.2], ms=3.5, label='<10 yr')
ax2.plot(aaa[d_p>365.25*10, 2], bbb[d_p>365.25*10, 2], 'o', c=[0.9, 0.9, 0.05], ms=3.5, label='<15 yr')
ax2.plot(aaa[d_p>365.25*15, 2], bbb[d_p>365.25*15, 2], 'o', c=[0.9, 0.35, 0.10], ms=3.5, label='<20 yr')
ax2.plot(aaa[d_p>365.25*20, 2], bbb[d_p>365.25*20, 2], 'o', c=[0.8, 0.15, 0.10], ms=3.5, label='>20 yr')

#ax2.plot(hector_E_list[:, 2], hector_N_list[:, 2], 'o', c='k', ms=4.7)
ax2.plot(np.arange(-1, 16, 1), np.arange(-1, 16, 1), ls='--',color='k', lw=3.5)
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

ax0.set_title("Kalman Horizontal Spectral Index", fontsize=36, pad=28)
ax0.set_xlabel("E-W component spectral index", fontsize=25, labelpad=14)
ax0.set_ylabel("N-S component spectral index", fontsize=25, labelpad=14)

ax1.set_title("Kalman Horizontal Color Noise Amplitude", fontsize=36, pad=28)
ax1.set_xlabel("E-W color noise amplitude", fontsize=25, labelpad=14)
ax1.set_ylabel("N-S color noise amplitude", fontsize=25, labelpad=14)

ax2.set_title("Kalman Horizontal White Noise Amplitude", fontsize=36, pad=28)
ax2.set_xlabel("E-W white noise amplitude", fontsize=25, labelpad=14)
ax2.set_ylabel("N-S white noise amplitude", fontsize=25, labelpad=14)

ax0.set_xlim(-2.8, 0.2)
ax0.set_ylim(-2.8, 0.2)

ax1.set_xlim(-0.5, 30)
ax1.set_ylim(-0.5, 30)

ax2.set_xlim(-0.5, 15)
ax2.set_ylim(-0.5, 15)

ax1.legend(fontsize=25)

ax0.set_aspect('equal')
ax1.set_aspect('equal')
ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig(outupt_path + "kf_hor_cor.png", dpi=400)


again = abs(abs(hector_E_list[:, 0]) - abs(hector_N_list[:, 0]))
again = [hector_list[i] for i, a in enumerate(again) if a>0.6]



fig = plt.figure(figsize=(12, 4))
ax0 = plt.subplot2grid((1, 1), (0, 0))

ax0.plot(d_p, hector_E_list[:, 0], 'o', ms=2)
ax0.set_ylim(-2.5, 0.2)


zz = hector_U_list[~np.isnan(aaa[:, 0]), 0]
tt = d_p[~np.isnan(aaa[:, 0])]

cc = np.array([3.5, 7.5, 12.5, 17.5, 22.5])

y = np.mean(zz[(tt>365.25*2) & (tt<365.25*5)])
yy = np.mean(zz[(tt>365.25*5) & (tt<365.25*10)])
yyy = np.mean(zz[(tt>365.25*10) & (tt<365.25*15)])
yyyy = np.mean(zz[(tt>365.25*15) & (tt<365.25*20)])
yyyyy = np.mean(zz[(tt>365.25*20)])

cccc = np.array([y, yy, yyy, yyyy, yyyyy])

plt.plot(cc, ccc, 'o',ms=4)
plt.plot(cc, cccc, 'o',ms=4)
