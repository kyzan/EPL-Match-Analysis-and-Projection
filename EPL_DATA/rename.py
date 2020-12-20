# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:09:42 2020

@author: Kartikay Sapra
"""

import os
for i in range(1,21):
    os.rename("E0 ("+str(i)+").csv",str(2020-i)+".csv")