# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:13:43 2024

@author: cdrg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def plot_wls_ts(time, gps_y, gps_y_sig, gps_full_decy, 
                wls_fit, offset_decy, name, component_str, output_path): 
    
    xmajor = 1
    xminor = 1/12
    
    if time[-1]-time[0]>10: 
        xmajor = 2
        xminor = 2/12
    
    if time[-1]-time[0]>20: 
        xmajor = 4
        xminor = 4/12
    
    fig0 = plt.figure(figsize=(14,8))
    ax0 = plt.subplot2grid((1,1),(0,0))
    
    ax0.errorbar(time, gps_y, gps_y_sig,
                 color="gray",alpha=0.38,fmt=".",zorder=1)
    
    ax0.plot(time, gps_y, "o", c="k", ms=2.5, zorder=2, label="raw data")
    ax0.plot(gps_full_decy, wls_fit, c="r", lw=2, zorder=3, label="WLS")
    
    if len(offset_decy):
        [ax0.axvline(a,linestyle='-.',c="blue",linewidth=1.4, zorder=2)
         for a in offset_decy]
    
    ax0.xaxis.set_major_locator(MultipleLocator(xmajor))
    ax0.xaxis.set_minor_locator(MultipleLocator(xminor))
    ax0.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    ax0.set_title(name+" (YZ-fixed) " + component_str, fontsize=30,pad=16)    
    ax0.set_xlabel("time in year",fontsize=18,labelpad=9)
    ax0.set_ylabel(component_str+" (mm)",fontsize=20,labelpad=9)
    
    ax0.tick_params(axis='both', which='major',direction='in',\
                            bottom=True,top=True,right=True,left=True,\
                            length=12, width=2.4, labelsize=16,pad=8.5)
    ax0.tick_params(axis='both', which='minor',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=6, width=1.8, labelsize=16,pad=8.5)
        
    [ax0.spines[a].set_linewidth(2.5) for a in ['top','bottom','left','right']]
    [ax0.spines[a].set_color("black") for a in ['top','bottom','left','right']]
    ax0.legend(fontsize=16)
    
    plt.tight_layout() 
    plt.savefig(output_path + name + "/wls_ts.png", dpi=300)
    plt.close("all")


def plot_wls_res(gps_full_decy, wls_res, freq_cpy, psd_lom, name, output_path, component_str):
    
    xmajor = 1
    xminor = 1/12
    
    if gps_full_decy[-1]-gps_full_decy[0]>10: 
        xmajor = 2
        xminor = 2/12
    
    if gps_full_decy[-1]-gps_full_decy[0]>20: 
        xmajor = 4
        xminor = 4/12
        
    fig2 = plt.figure(figsize=(17,10))
    ax0 = plt.subplot2grid((2,2),(0,0),colspan=2)
    ax1 = plt.subplot2grid((2,2),(1,0),colspan=1)
    ax2 = plt.subplot2grid((2,2),(1,1))
    axes = [ax0, ax1, ax2]
    
    ax0.plot(gps_full_decy, wls_res, color="black", lw=0.33)
    ax0.plot(gps_full_decy, wls_res,"o", color="black", ms=2.9)
    ax0.yaxis.set_minor_locator(AutoMinorLocator(5))  
    
    ax1.hist(wls_res[~np.isnan(wls_res)], bins=20, rwidth=0.88, color = "steelblue", alpha=0.9)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))  
    
    ax2.loglog(freq_cpy ,psd_lom, color="black", lw=1.1)    

    ax0.set_title(name + " (YZ-fixed) " + component_str + " WLS Residual", fontsize=34, pad=10)
    ax1.set_title("Residual Histogram", fontsize=28, pad=10)
    ax2.set_title("Residual PSD", fontsize=28, pad=10)

    ax0.set_xlabel("time in year",fontsize=23, labelpad=10)
    ax0.set_ylabel("residual (mm)",fontsize=23, labelpad=10)
    ax1.set_xlabel("residual (mm)",fontsize=23, labelpad=10)
    ax1.set_ylabel("data points",fontsize=23, labelpad=10)
    ax2.set_xlabel("frequency (cpy)",fontsize=23, labelpad=10)
    ax2.set_ylabel("PSD (mm$^2$/cpy)",fontsize=23, labelpad=10)

    ax0.xaxis.set_major_locator(MultipleLocator(xmajor))
    ax0.xaxis.set_minor_locator(MultipleLocator(xminor))
    
    [ax.spines[a].set_linewidth(2.35) for a in ['top','bottom','left','right']  for ax in axes]
    [ax.spines[a].set_color("black") for a in ['top','bottom','left','right'] for ax in axes]
    
    ax0.tick_params(axis='both', which='major',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=12, width=2.4, labelsize=17.5, pad=8)
        
    ax0.tick_params(axis='both', which='minor',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=6, width=1.8, labelsize=17.5, pad=8)
        
    ax1.tick_params(axis='both', which='major',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=10, width=2, labelsize=17.5, pad=8)
        
    ax1.tick_params(axis='both', which='minor',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=5, width=1.5, labelsize=17.5, pad=8)
        
    ax2.tick_params(axis='both', which='major',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=7, width=1.8, labelsize=17.5, pad=8)
        
    ax2.tick_params(axis='both', which='minor',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=3.5, width=1, labelsize=17.5, pad=8)

    plt.tight_layout() 
    plt.savefig(output_path + name + "/wls_res.png", dpi=300)
    plt.close("all")

    
def plot_wls_comp(gps_full_decy, offset_decy, wls_trend, wls_annual, wls_semi_annual, 
                  wls_res, name, output_path, component_str):
    
    xmajor = 1
    xminor = 1/12
    
    if gps_full_decy[-1]-gps_full_decy[0]>10: 
        xmajor = 2
        xminor = 2/12
    
    if gps_full_decy[-1]-gps_full_decy[0]>20: 
        xmajor = 4
        xminor = 4/12
        
    fig5 = plt.figure(figsize=(12,17))
    ax0 = plt.subplot2grid((4,1),(0,0))
    ax1 = plt.subplot2grid((4,1),(1,0))
    ax2 = plt.subplot2grid((4,1),(2,0))
    ax3 = plt.subplot2grid((4,1),(3,0))
    
    axes = [ax0,ax1,ax2,ax3]
    
    ax0.plot(gps_full_decy, wls_trend, c="k", lw=2)
    
    if len(offset_decy):
        [ax0.axvline(a,linestyle='-.',c="blue",linewidth=1.4) \
         for a in offset_decy]
            
    ax0.yaxis.set_minor_locator(AutoMinorLocator(5))                
    ax1.plot(gps_full_decy,wls_annual,c="black",linewidth=2) 
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax2.plot(gps_full_decy,wls_semi_annual,c="black",linewidth=2)
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax3.plot(gps_full_decy, wls_res, c="k", linewidth=0.5)
    ax3.plot(gps_full_decy, wls_res, 'o', c="k", ms=3.5)
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5))    
        
    [ax.xaxis.set_major_locator(MultipleLocator(xmajor)) for ax in axes]
    [ax.xaxis.set_minor_locator(MultipleLocator(xminor)) for ax in axes]
         
    [ax.tick_params(axis='both', which='major',direction='in',\
            bottom=True,top=True,right=True,left=True,\
            length=12, width=2.4, labelsize=18, pad=10) \
             for ax in axes]
    
    [ax.tick_params(axis='both', which='minor',direction='in',\
            bottom=True,top=True,right=True,left=True,\
            length=6, width=1.8, labelsize=18, pad=10)\
             for ax in axes]        
    
    [ax.spines[a].set_linewidth(2.7) for a in ['top','bottom','left','right'] for ax in axes]   
    [ax.spines[a].set_color("black") for a in ['top','bottom','left',"right"] for ax in axes]        
    
    ax0.set_title(name+" (YZ-fixed) "+component_str+" WLS",fontsize=32,pad=15)
    
    ax0.set_ylabel("trend (mm)",fontsize=22,labelpad=8)
    ax1.set_ylabel("annual (mm)",fontsize=22,labelpad=8)
    ax2.set_ylabel("semi-annual (mm)",fontsize=22,labelpad=8)
    ax3.set_ylabel("residual (mm)",fontsize=22,labelpad=8)
    ax3.set_xlabel("time in year",fontsize=22,labelpad=8)
    
    plt.tight_layout(); 
    plt.savefig(output_path + name + "/wls_comp.png", dpi=300)
    
    plt.close("all")
    
    


