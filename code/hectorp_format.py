# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:03:09 2023

@author: cdrg
"""
import os
import numpy as np
from decimal_year2date import decimal_year2date

#+=========================================================
#import gps data
gps_path = "D:/RA_all/taiwan_all/output/old_format/"
gps_list = os.listdir(gps_path)

direction = 1
if direction==1: direction_str="E"
elif direction==2: direction_str="N"
elif direction==3: direction_str="U"

#offset data
offset_path = r"D:/RA_all/GS_stations_noise/data/gps_timeseries/IGb14/"
offset_list = os.listdir(offset_path)
output_path = "D:/RA_all/taiwan_all_ls_post/output/hector/ori/"
ctl_path = "D:/RA_all/taiwan_all_ls_post/output/hector/ctl/"

post_path = "D:/RA_all/wls_post/output/post_const/"
if direction==1 : post_path=post_path+"E-W/"
if direction==2 : post_path=post_path+"N-S/"
if direction==3 : post_path=post_path+"U-D/"

post_list = os.listdir(post_path)

#+=========================================================
def compute_mjd(date):
    date_list =list(map(int,date.split("/")))
    [year, month, day, hour, minute, second] = date_list

    mjd = 367*year - int(7*(year+int((month+9)/12))/4) + \
            int(275*month/9) +  day + 1721014 - 2400001
    mjd += (hour + (minute + second/60.0)/60.0)/24.0
    return mjd

#+=========================================================

name_list = []
for gps_file in gps_list: 
    name, ext = os.path.splitext(gps_file) 
    '''
    if name not in ["YILN", "DASI", "GFPS", "GS28", "HERI", "HLIU", "HSR4",
                    "I10R", "KFN2", "NIUT", "NPRS", "PUSN", "SNES", "SUN1"
                    "SPAO", "SUAN", "SUCH", "WHES", "T103", "T104", "TCMS", "GMRS"]:continue
    '''
    if name not in ["SUNM"]:continue
    if ext == '.txt':
        name = name.split(".")[0]
        name_list.append(name)
        
        with open(gps_path+gps_file) as file:
            gps = []
            for line in file:
                gps.append(line.rstrip().split())
                                
        del gps[0],gps[0],file,line
        
        gps = np.array(gps,dtype=float)
        
        gps_time = gps[:,0]
        gps_pos = gps[:,direction]*1000
        gps_sig = (gps[:,direction+3]*1000)
        
        date = [''.join(["/"+str(int(b)) if i>0 else str(int(b))\
         for i,b in enumerate(a[11::])]) for a in gps]
            
        mjd_raw = [compute_mjd(a) for a in date]  
        
        #============================================
        #offset
        file_str = gps_file.split(".")
        new_file_format = file_str[0]+"."+"break" 
        offset_mjd = []
        if new_file_format in offset_list:
            with open(offset_path+new_file_format) as file:
                offset_time = []
                for line in file:
                    offset_time.append(line.rstrip().split()[1:7])
            
            offset_date = [''.join(["/"+str(int(b)) if i>0 else str(int(b))\
                                    for i,b in enumerate(a)]) for a in offset_time]
            
            if ("2022/9/17/13/41/19" in offset_date) and("2022/9/18/6/44/15" in offset_date):
                offset_date.remove("2022/9/18/6/44/15")
                
            offset_mjd = [compute_mjd(a) for a in offset_date]
        
        #============================================
        # post       
        post_mjd = []
        if gps_file in post_list:
            with open(post_path+gps_file) as file:
                post_time = []
                for line in file:
                    post_time.append(line.rstrip().split())
                    
            post_time, post_const = np.array(post_time[1::], dtype = float).T    
            post_const = np.array(post_const)
            
            post_date = decimal_year2date(post_time)
            post_mjd = [compute_mjd(a) for a in post_date]
            
        
        #============================================
        #output
        with open (output_path+direction_str+"/"+name+".mom","w") as f:
            f.write("# sampling period 1.0\n")
            if len(offset_mjd):
                [f.write("# offset "+str(a)+"\n") for a in offset_mjd]
            
            if len(post_mjd):
                [f.write("# log "+str(a)+" "+str(b)+"\n") for a, b in zip(post_mjd, post_const)]
            
            [f.write(" "+str(a)+" "+str(round(b,1))+"\n") for a,b in zip(mjd_raw,gps_pos)]
        
        #ctl file
        with open (ctl_path+direction_str+"/"+name+".ctl","w") as f:
            f.write("DataFile            "+name+".mom\n")
            f.write("DataDirectory       "+output_path+direction_str+"/\n")
            f.write("OutputFile          "+"D:/RA_all/taiwan_all_ls_post/output/hector/output/"+direction_str+"/"+name+".mom"+"\n")
            f.write("interpolate         no\n")
            f.write("PhysicalUnit        mm\n")
            f.write("ScaleFactor         1.0\n")
            f.write("periodicsignals     365.25 182.625\n")
            f.write("estimateoffsets     yes\n")
            f.write("NoiseModels         GGM White\n")
            f.write("GGM_1mphi           6.9e-06\n")
            f.write("useRMLE             no\n")
            f.write("Verbose             yes")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        