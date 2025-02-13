# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 01:55:44 2020

@author: user
"""

from datetime import datetime, timedelta
import numpy as np

def decimal_year2date(time):

  new_time = []
  for i  in range(len(time)):

    start = time[i]
    year = int(start)
    rem = start - year

    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
    
    year_ = result.year
    month_ = result.month
    day_ = result.day
    
    hour_ = result.hour
    min_ = result.minute
    s_ = result.second
    
    a = str(year_)+"/"+str(month_)+"/"+str(day_)+"/"+str(hour_)+"/"+str(min_)+"/"+str(s_)
    new_time.append(a)
    
  return(new_time)

