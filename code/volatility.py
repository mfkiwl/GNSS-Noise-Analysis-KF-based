# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:44:22 2024

@author: cdrg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

#==================================================
GPS_DIR = "D:/RA_all/taiwan_all_ls_post/output/kf_ar3/N-S/"
gps_list = os.listdir(GPS_DIR)
data_num = len(gps_list)

pos_path = "D:/RA_all/GS_stations_noise/data/"
pos_filename = "gps_pos.txt"

gps_info_dir = r"D:/RA_all/taiwan_all/"
gps_info_filename = "gps_mon.txt"

dome_dir =  r"D:/RA_all/GS_stations_noise/data/"
dome_filename = "dome_new.txt"

#==================================================
def import_data(data_dir, filename):
    data = []
    with open(data_dir + filename, encoding="utf-8") as file:
        for line in file:
            data.append(line.strip().split())
            
    return data


def get_monu_type(data):
    monu_dict = {"" : 0, 
                 "pillarconcrete" : 1, 
                 "roof" : 2, 
                 "metalwra" : 3, 
                 "rodmetal" : 4, 
                 "ddb" : 5,
                 "sdb" : 6}
    
    return monu_dict[data[1]]


#==================================================
pos = import_data(pos_path, pos_filename)[1::]
pos_name = [a[0] for a in pos]
pos_arr = np.zeros([data_num, 3])

# monu
gps_info_data = import_data(gps_info_dir, gps_info_filename)
gps_mon_data = [[a[0], ""] if len(a)==6 else [a[0], a[-2].lower()]\
                if len(a)==7 else [a[0], a[-2].lower() + a[-3].lower()]\
                for a in gps_info_data]

gps_mon_data = [a for a in gps_mon_data if len(a[1])!=0]    
gps_mon_name = [a[0] for a in gps_mon_data]  
 
gps_type = [a for a in list(set([a[1] for a in gps_mon_data]))]


# dome
dome = import_data(dome_dir, dome_filename)
dome_name = [a[0] for a in dome]

phase_all = []
noise_phase_all = []
noise_amp_all = []
sea_amp_all = []

name_list = []
monu_list = []
dome_list = []

pos_list = []
for i, st in enumerate(gps_list):
    if st in ["CHIN", "DANI", "DASI", "DSIN", "ERLN", 
              "FENP", "FKDO", "GAIS", "GS10", "GS22",
              "GS28", "GS43", "GS47", "GS66", "GS68", 
              "GS69", "GS74", "GS94", "GS95", "HCHM",
              "HGC1", "HOBE", "HOPN", "KSH2", "LIKN", 
              "LIUC", "LIYU", "MITO", "MUDA", "NDH9",
              "PUSN", "SAME", "SANI", "SEED", "SHLU",
              "SNES", "SUAN", "TASI", "TATA", "TASO", 
              "W021", "W029", "WIAN", "YILN"] : continue
    
    name = st[0:4]
    if name not in gps_mon_name:continue   
    
    ind = gps_mon_name.index(name)
    if len(gps_mon_data[ind][1])==0:continue
    
    mu = np.array(import_data(GPS_DIR + st + "/", "rts_mu.txt"), dtype=float)
    H = np.array(import_data(GPS_DIR + st + "/", "H.txt"), dtype=float)
    
    #if len(mu)<365.25*15:continue
    
    noise = np.array(import_data(GPS_DIR + st + "/", "noise.txt")[2::], dtype=float)
    
    nan_ind = np.isnan(noise[:, -1])
    white = np.abs(noise[~nan_ind, -1])
    white = white - np.mean(white)
    
    
    time = noise[~nan_ind, 0]

    G = np.array([[np.sin(2*np.pi*t), np.cos(2*np.pi*t)]
                   for t in time])
    
    m = np.linalg.lstsq(G, white, rcond=None)[0]
    noise_amp_all.append(np.sqrt(m[0]**2 + m[1]**2))
    
    noise_phase = np.degrees(np.arctan2(m[0], m[1])) + 180
    noise_phase_all.append(noise_phase)

    # - cos only -> phi = 0
    sin = mu[:, 3]
    cos = mu[:, 4]
    
    monu_num = get_monu_type(gps_mon_data[ind])
    name_list.append(name)
    monu_list.append(monu_num)
    sea_amp_all.append(np.sqrt(np.mean(sin)**2+np.mean(cos)**2))
    
    phase = np.degrees(np.arctan2(sin, cos))
    mean_phase = np.mean(phase) + 180
    phase_all.append(mean_phase)
    
    if name in pos_name:
        if name in dome_name:
            pos_ind = pos_name.index(name)
            pos_arr[i] = pos[pos_ind][1::]
            pos_list.append(pos[pos_ind][1::])
            
            dome_ind = dome_name.index(name)
            dome_list.append(dome[dome_ind][1])

    '''
    fig = plt.figure(figsize=(14,8))
    plt.plot(time, white)
    plt.plot(time, G@m)
    '''
    
    #break

noise_amp_all = np.array(noise_amp_all)
phase_all = np.array(phase_all)
pos_list = np.array(pos_list, dtype=float)
sea_amp_all = np.array(sea_amp_all, dtype=float)
#==================================================
fig = plt.figure(figsize=(12, 8))
ax0 = plt.subplot2grid((1, 1), (0, 0))

ax0.hist(phase_all, 12, rwidth = 0.88)
ax0.set_xlim(0, 360)



fig = plt.figure(figsize=(12, 8))
ax0 = plt.subplot2grid((1, 1), (0, 0))

ax0.hist(noise_phase_all, 12, rwidth = 0.88)
ax0.set_xlim(0, 360)
 

#ax0.plot(np.abs(noise_phase_all), np.abs(phase_all), 'o', c='k', ms=2)
    
phase_all = np.array([np.abs(a-180)  for a in phase_all])
noise_phase_all = np.array([np.abs(a-180)  for a in noise_phase_all])
    
from mpl_toolkits.basemap import Basemap 
fig = plt.figure(figsize=(14,14))

ma = Basemap(projection='cyl',resolution="i",llcrnrlat=21.74, urcrnrlat=25.5, llcrnrlon=119.8, urcrnrlon=122.1)
ma.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', \
              service='World_Shaded_Relief', xpixels=400, ypixels=None, dpi=200, verbose=False)  
  
aa = plt.scatter(pos_list[:, 1], pos_list[:, 0], c=phase_all,s=100, cmap='afmhot', \
            vmin = 0, vmax = 180)

color_bar = fig.colorbar(aa, extend = 'both') 
plt.tight_layout()

    
#==================================================
fig = plt.figure(figsize=(14,14))

ma = Basemap(projection='cyl',resolution="i",llcrnrlat=21.74, urcrnrlat=25.5, llcrnrlon=119.8, urcrnrlon=122.1)
ma.arcgisimage(server='http://server.arcgisonline.com/ArcGIS',
              service='World_Shaded_Relief', xpixels=400, ypixels=None, dpi=200, verbose=False)  

aa = plt.scatter(pos_list[:, 1], pos_list[:, 0], c=noise_phase_all, s=100, cmap='afmhot',
            vmin = 0, vmax = 180)

color_bar = fig.colorbar(aa, extend = 'both')
plt.tight_layout()




#==================================================
fig = plt.figure(figsize=(14,14))

ma = Basemap(projection='cyl',resolution="i",llcrnrlat=21.74, urcrnrlat=25.5, llcrnrlon=119.8, urcrnrlon=122.1)
ma.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', \
              service='World_Shaded_Relief', xpixels=400, ypixels=None, dpi=200, verbose=False)  
  
aa = plt.scatter(pos_list[:, 1], pos_list[:, 0], c=noise_amp_all
                 , s=100, cmap='afmhot_r', vmin = 0, vmax = 0.3)

color_bar = fig.colorbar(aa, extend = 'both') 
plt.tight_layout()      
 


monu_arr = np.array(monu_list, dtype=float)

pill_ind = monu_arr==1
roof_ind = monu_arr==2
metal_ind = monu_arr==3
rod_ind = monu_arr==4
ddb_ind = monu_arr==5
sdb = monu_arr==6

all_ind = [pill_ind, roof_ind, metal_ind,\
           rod_ind, ddb_ind, sdb]
    
all_str = ["Concrete Pillar", "Roof", "WRA", 
          "Metal Rod", "Deep Drilled Braced", "Shallow Drilled Braced"]


fig = plt.figure(figsize=(12, 20))
ax0 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
ax1 = plt.subplot2grid((4, 2), (1, 0))
ax2 = plt.subplot2grid((4, 2), (2, 0))
ax3 = plt.subplot2grid((4, 2), (3, 0))
ax4 = plt.subplot2grid((4, 2), (1, 1))
ax5 = plt.subplot2grid((4, 2), (2, 1))
ax6 = plt.subplot2grid((4, 2), (3, 1))
axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]

ax0.hist(noise_amp_all, bins=np.arange(0, 2, 0.1), rwidth=0.9)
ax0.yaxis.set_minor_locator(AutoMinorLocator(5))  

for ax, ind_type, str_ in zip(axes[1::], all_ind, all_str):
    ax.hist(noise_amp_all[ind_type], bins=np.arange(0, 2, 0.1), rwidth=0.88)
    # ax.hist(phase_all[ind_type], bins=np.arange(0, 360, 10), rwidth=0.88)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    ax.set_title(str_,fontsize=26,pad=15)
    ax.set_xlabel("spectral index",fontsize=22,labelpad=12)
    ax.set_ylabel("station number",fontsize=22,labelpad=12)

[ax.xaxis.set_major_locator(MultipleLocator(0.5)) for ax in axes]
[ax.xaxis.set_minor_locator(MultipleLocator(0.1)) for ax in axes]
# [ax.xaxis.set_major_locator(MultipleLocator(90)) for ax in axes]
# [ax.xaxis.set_minor_locator(MultipleLocator(18)) for ax in axes]

[ax.tick_params(axis='both', which='major',direction='in',
                bottom=True,top=True,right=True,left=True,
                length=10, width=2, labelsize=18,pad=10) for ax in axes]

[ax.tick_params(axis='both', which='minor',direction='in',
                bottom=True,top=True,right=True,left=True,
                length=5, width=1.5, labelsize=18,pad=10) for ax in axes]
    
[ax.spines[b].set_linewidth(2) for b in ['top','bottom','left','right'] for ax in axes]
[ax.spines[b].set_color("black") for b in ['top','bottom','left','right'] for ax in axes]

ax0.set_title("Hector E-W Spectral Index",fontsize=30,pad=15)    
ax0.set_xlabel("spectral index",fontsize=22,labelpad=12)
ax0.set_ylabel("station number",fontsize=22,labelpad=12)

#ax0.set_ylim(0, 90)

plt.tight_layout()   




fig = plt.figure(figsize=(13, 7))
ax0 = plt.subplot2grid((1, 1), (0, 0))

ax0.plot(pos_list[noise_amp_all<2, 2], noise_amp_all[noise_amp_all<2], 'o', ms=3)


'''
dome_str = {'AUST':1,
            'CONE':2,
            'DOME':3,
            'GSI':4,
            'JAVC':5,
            'JPLA':6,
            'LEIM':8,
            'LEIS':9,
            'LEIT':10,
            'NONE':11,
            'SCIS':12,
            'SCIT':13,
            'SNOW':14,
            'TPSH':15,
            'UNAV':16,
            'UNKN':17}
'''
dome_str = {'CONE' : 1, 'DOME' : 2, 'JAVC' : 3, 'LEIS' : 4,
            'NONE' : 5, 'SCIS' : 6, 'SCIT' : 7, 'TPSH' : 8,
            'UNKN' : 9}


dome_type = [a for a in dome_list]
dome_type_int = np.array([dome_str[a] for a in dome_list], dtype=int)


fig = plt.figure(figsize=(13, 7))
ax0 = plt.subplot2grid((1, 1), (0, 0))

ax0.hist(phase_all[dome_type_int==7], bins=np.arange(0, 180, 10), rwidth=0.88)

