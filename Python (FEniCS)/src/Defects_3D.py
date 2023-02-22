
# Define class 
from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from ufl.operators import Dt 
# Create mesh and define function space
#%matplotlib notebook 
from tqdm import tqdm
import os 
import shutil
import os
from base64 import decodebytes

from sample_script import *


class Defects:
    ''' Process the gmesh file into data array, and export into a .dat format'''
    def __init__(self):
        pass


    def random_init(self):
        S_int_2_a = Expression('x[0]*x[0]+x[1]*x[1]+x[2]*x[2]', degree=2)

        return S_int_2_a 
    
    def pole_init(self,h):
        S_int_2_a = Expression('(sqrt(x[2]*x[2]) < h) ? 4:0.01', h=h, degree=2)

        return S_int_2_a 

    def two_defect (self,x0,y0,x3=0,y3=0, charge_1=1,charge_2=1 ,direction_angle_1=0,direction_angle_2=0 ):
        S_int_2_a = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ) ,x0=-x3,y0=y3,x1=-x0,y1=y0 ,charge_1=charge_1,charge_2=charge_2 ,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2 ,core_size=1, degree=1)

        S_int_2_b = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
              )  ,x0=-x3,y0=y3,x1=-x0,y1=y0 ,charge_1=charge_1,charge_2=charge_2 ,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2  ,core_size=1, degree=1)

    def single_defect (self,x0,y0, charge_1=1,direction_angle_1=0 ):
        S_int_2_a = Expression(\
            ('cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ) ,x0=x0,y0=y0 ,charge_1=charge_1,direction_angle_1=direction_angle_1,core_size=1, degree=1)


        return S_int_2_a