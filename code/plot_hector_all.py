# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:04:37 2024

@author: cdrg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:57:42 2024

@author: cdrg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

#=========================================================
#init
hector_path_E = "D:/RA_all/taiwan_all_ls_post/output/hector/output/E/"
hector_path_N = "D:/RA_all/taiwan_all_ls_post/output/hector/output/N/"
hector_path_U = "D:/RA_all/taiwan_all_ls_post/output/hector/output/U/"
hector_list = [a for a in os.listdir(hector_path_E) if a[-3::] == 'txt']

data_num = len(hector_list)
outupt_path = "D:/RA_all/taiwan_all_ls_post/output/pic/"
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

d_p = []
for i, st in enumerate(hector_list):
    
    hector_data_E = [a for a in import_data(hector_path_E, st) if len(a)!=0]
    hector_data_N = [a for a in import_data(hector_path_N, st) if len(a)!=0]
    hector_data_U = [a for a in import_data(hector_path_U, st) if len(a)!=0]
    
    hector_k_E = [a for a in hector_data_E if a[0]=="kappa"]
    hector_amp_E = [a for a in hector_data_E if a[0]=="sigma"]
    
    hector_k_N = [a for a in hector_data_N if a[0]=="kappa"]
    hector_amp_N = [a for a in hector_data_N if a[0]=="sigma"]
    
    hector_k_U = [a for a in hector_data_U if a[0]=="kappa"]
    hector_amp_U = [a for a in hector_data_U if a[0]=="sigma"]
    
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
fig.subplots_adjust(top=0.8)

hector_E_mean = np.mean(hector_E_list[~np.isnan(hector_E_list[:, 0])], axis=0)
hector_N_mean = np.mean(hector_N_list[~np.isnan(hector_E_list[:, 0])], axis=0)
hector_U_mean = np.mean(hector_U_list[~np.isnan(hector_E_list[:, 0])], axis=0)

ax0.hist(hector_E_list[~np.isnan(hector_E_list[:, 0]), 0], bins=np.arange(-3, 0.01, 0.1), rwidth=0.85, color=[0.2, 0.2, 0.2])
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

plt.suptitle("Hector Noise Analysis", fontsize=40, y=0.98)
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

ax1.set_ylim(0, 280)
ax4.set_ylim(0, 280)
ax7.set_ylim(0, 280)

ax2.set_ylim(0, 320)
ax5.set_ylim(0, 320)
ax8.set_ylim(0, 320)
plt.tight_layout(pad=3.5)
plt.savefig(outupt_path + "hector_hist.png", dpi=400)


d_p = np.array(d_p)
fig = plt.figure(figsize=(36, 13))
ax0 = plt.subplot2grid((1, 3), (0, 0))
ax1 = plt.subplot2grid((1, 3), (0, 1))
ax2 = plt.subplot2grid((1, 3), (0, 2))
axes = [ax0, ax1, ax2]

aaa = hector_E_list.copy()
bbb = hector_N_list.copy()

ax0.plot(aaa[d_p>365.25*2, 0], bbb[d_p>365.25*2, 0], 'o', c=[0.2, 0.2, 0.8], ms=8.5, label='<5 yr')
ax0.plot(aaa[d_p>365.25*5, 0], bbb[d_p>365.25*5, 0], 'o', c=[0.5, 0.7, 0.2], ms=8.5, label='<10 yr')
ax0.plot(aaa[d_p>365.25*10, 0], bbb[d_p>365.25*10, 0], 'o', c=[0.9, 0.9, 0.05], ms=8.5, label='<15 yr')
ax0.plot(aaa[d_p>365.25*15, 0], bbb[d_p>365.25*15, 0], 'o', c=[0.9, 0.35, 0.10], ms=8.5, label='<20 yr')
ax0.plot(aaa[d_p>365.25*20, 0], bbb[d_p>365.25*20, 0], 'o', c=[0.8, 0.15, 0.10], ms=8.5, label='>20 yr')

#ax0.plot(hector_E_list[:, 0], hector_N_list[:, 0], 'o', c='k', ms=6.6)
ax0.plot(np.arange(-4, 1, 0.1), np.arange(-4, 1, 0.1), ls='--',color='k', lw=3.5)
ax0.yaxis.set_minor_locator(AutoMinorLocator(5))
ax0.xaxis.set_minor_locator(AutoMinorLocator(5))


ax1.plot(aaa[d_p>365.25*2, 1], bbb[d_p>365.25*2, 1], 'o', c=[0.2, 0.2, 0.8], ms=6, label='<5 yr')
ax1.plot(aaa[d_p>365.25*5, 1], bbb[d_p>365.25*5, 1], 'o', c=[0.5, 0.7, 0.2], ms=6, label='<10 yr')
ax1.plot(aaa[d_p>365.25*10, 1], bbb[d_p>365.25*10, 1], 'o', c=[0.9, 0.9, 0.05], ms=6, label='<15 yr')
ax1.plot(aaa[d_p>365.25*15, 1], bbb[d_p>365.25*15, 1], 'o', c=[0.9, 0.35, 0.10], ms=6, label='<20 yr')
ax1.plot(aaa[d_p>365.25*20, 1], bbb[d_p>365.25*20, 1], 'o', c=[0.8, 0.15, 0.10], ms=6, label='>20 yr')

#ax1.plot(hector_E_list[:, 1], hector_N_list[:, 1], 'o', c='k', ms=4.7)
ax1.plot(np.arange(-1, 51, 1), np.arange(-1, 51, 1), ls='--',color='k', lw=3.5)
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))


ax2.plot(aaa[d_p>365.25*2, 2], bbb[d_p>365.25*2, 2], 'o', c=[0.2, 0.2, 0.8], ms=4.5, label='<5 yr')
ax2.plot(aaa[d_p>365.25*5, 2], bbb[d_p>365.25*5, 2], 'o', c=[0.5, 0.7, 0.2], ms=4.5, label='<10 yr')
ax2.plot(aaa[d_p>365.25*10, 2], bbb[d_p>365.25*10, 2], 'o', c=[0.9, 0.9, 0.05], ms=4.5, label='<15 yr')
ax2.plot(aaa[d_p>365.25*15, 2], bbb[d_p>365.25*15, 2], 'o', c=[0.9, 0.35, 0.10], ms=4.5, label='<20 yr')
ax2.plot(aaa[d_p>365.25*20, 2], bbb[d_p>365.25*20, 2], 'o', c=[0.8, 0.15, 0.10], ms=4.5, label='>20 yr')

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

ax0.set_title("Hector Horizontal Spectral Index", fontsize=36, pad=28)
ax0.set_xlabel("E-W component spectral index", fontsize=25, labelpad=14)
ax0.set_ylabel("N-S component spectral index", fontsize=25, labelpad=14)

ax1.set_title("Hector Horizontal Color Noise Amplitude", fontsize=36, pad=28)
ax1.set_xlabel("E-W color noise amplitude", fontsize=25, labelpad=14)
ax1.set_ylabel("N-S color noise amplitude", fontsize=25, labelpad=14)

ax2.set_title("Hector Horizontal White Noise Amplitude", fontsize=36, pad=28)
ax2.set_xlabel("E-W white noise amplitude", fontsize=25, labelpad=14)
ax2.set_ylabel("N-S white noise amplitude", fontsize=25, labelpad=14)

ax0.set_xlim(-2.8, 0.2)
ax0.set_ylim(-2.8, 0.2)

ax1.set_xlim(-0.5, 30)
ax1.set_ylim(-0.5, 30)

ax2.set_xlim(-0.5, 15)
ax2.set_ylim(-0.5, 15)

ax0.set_aspect('equal')
ax1.set_aspect('equal')
ax2.set_aspect('equal')

ax1.legend(fontsize=25)

plt.tight_layout()
plt.savefig(outupt_path + "hector_hor_color.png", dpi=400)




aaa[d_p>365.25*2, 0], bbb[d_p>365.25*2, 0]
aaa[d_p>365.25*5, 0], bbb[d_p>365.25*5, 0]
aaa[d_p>365.25*10, 0], bbb[d_p>365.25*10, 0]
aaa[d_p>365.25*15, 0], bbb[d_p>365.25*15, 0]
aaa[d_p>365.25*20, 0], bbb[d_p>365.25*20, 0]

zz = hector_U_list[~np.isnan(aaa[:, 0]), 0]
tt = d_p[~np.isnan(aaa[:, 0])]

cc = np.array([3.5, 7.5, 12.5, 17.5, 22.5])

x = np.mean(zz[(tt>365.25*2) & (tt<365.25*5)])
xx = np.mean(zz[(tt>365.25*5) & (tt<365.25*10)])
xxx = np.mean(zz[(tt>365.25*10) & (tt<365.25*15)])
xxxx = np.mean(zz[(tt>365.25*15) & (tt<365.25*20)])
xxxxx = np.mean(zz[(tt>365.25*20)])


ccc = np.array([x, xx, xxx, xxxx, xxxxx])
plt.plot(cc, ccc, 'o')
