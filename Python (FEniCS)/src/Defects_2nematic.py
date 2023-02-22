
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

    def single_defect_zero_wnt(self,x0=0,y0=0,x2=0,y2=0,direction_angle=3.14,charge=1,direction_angle_2=3.14,charge_2=1,z=5):
        S_int_1 = Expression(
            ('sqrt(2)*cos(charge*(atan2(x[1]-y0,x[0]-x0)+direction_angle))*(core_size)*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))',
            'sqrt(2)*sin(charge*(atan2(x[1]-y0,x[0]-x0)+direction_angle))*(core_size)*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))',
            'sqrt(2)*cos(charge_2*(atan2(x[1]-y2,x[0]-x2)+direction_angle_2))*(core_size)*sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*sqrt((0.34+0.07*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)))*1/(1+0.41*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))+0.07*(core_size)*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))))',
            'sqrt(2)*sin(charge_2*(atan2(x[1]-y2,x[0]-x2)+direction_angle_2))*(core_size)*sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*sqrt((0.34+0.07*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)))*1/(1+0.41*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))+0.07*(core_size)*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))))',
            '((sqrt(x[0]*x[0]+x[1]*x[1])<z))  ?(1/z)*(z-(sqrt(x[0]*x[0]+x[1]*x[1]))):0.01'),z=z,x0=x0,y0=y0,x2=x2,y2=y2,direction_angle=direction_angle,direction_angle_2=direction_angle_2,core_size=1,charge=charge,charge_2=charge_2, degree=1)

        return S_int_1

    def two_defect_center_wnt(self,x0,y0,x3=0,y3=0,z=5,x_bot1=0,y_bot1=0,x_bot2=0,y_bot2=0,charge_1=1,charge_2=1,charge_3=0,charge_4=0,direction_angle_1=0,direction_angle_2=0,direction_angle_3=0,direction_angle_4=0):
        S_int_2_a = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
                ,'((sqrt(x[0]*x[0]+x[1]*x[1])<z))  ?(1/z)*(z-(sqrt(x[0]*x[0]+x[1]*x[1]) ) ):0.01'),z=z,A=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        S_int_2_b = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2)))*1/(1+0.41*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))+0.07*(core_size)*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2)))*1/(1+0.41*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))+0.07*(core_size)*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))))'\
            ,'((sqrt(x[0]*x[0]+x[1]*x[1])<z))  ?(1/z)*(z-(sqrt(x[0]*x[0]+x[1]*x[1]))):0.01'),z=z,A=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        return S_int_2_a+S_int_2_b
        
    def two_defect_ellipse_wnt(self,x0,y0,x3=0,y3=0,z_1=6,z_2=4,x_bot1=0,y_bot1=0,x_bot2=0,y_bot2=0,charge_1=1,charge_2=1,charge_3=0,charge_4=0,direction_angle_1=0,direction_angle_2=0,direction_angle_3=0,direction_angle_4=0):
        z=(z_1*z_2)
        S_int_2_a = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
                ,'(( sqrt(x[0]*x[0]+x[1]*x[1])<=z_2))  ?(0.5/z)*(-(z_1*z_2/sqrt(1+ (z_2*z_2/z_1)*(z_2*z_2/z_1)+(z_1*z_2/z_2)*(z_1*z_2/z_2) ) )+ (z_1*z_2/sqrt(1+ (z_2*x[0]/z_1)*(z_2*x[0]/z_1)+(z_1*x[1]/z_2)*(z_1*x[1]/z_2) ) ) ):0.01'),z=z,z_1=z_1,z_2=z_2,A=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        S_int_2_b = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2)))*1/(1+0.41*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))+0.07*(core_size)*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2)))*1/(1+0.41*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))+0.07*(core_size)*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))))'\
            ,'(( sqrt(x[0]*x[0]+x[1]*x[1])<=z_2))  ? (0.5/z)*(-(z_1*z_2/sqrt(1+ (z_2*z_2/z_1)*(z_2*z_2/z_1)+(z_1*z_2/z_2)*(z_1*z_2/z_2) ) )+(z_1*z_2/sqrt(1+ (z_2*x[0]/z_1)*(z_2*x[0]/z_1)+(z_1*x[1]/z_2)*(z_1*x[1]/z_2) ) ) ):0.01'),z=z,z_1=z_1,z_2=z_2,A=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        return S_int_2_a+S_int_2_b

    def single_defect_two_layer(self,x0=0,y0=0,x2=0,y2=0,direction_angle=3.14,charge=1,direction_angle_2=3.14,charge_2=1):
        S_int_1 = Expression(
            ('sqrt(2)*cos(charge*(atan2(x[1]-y0,x[0]-x0)+direction_angle))*(core_size)*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))',
            'sqrt(2)*sin(charge*(atan2(x[1]-y0,x[0]-x0)+direction_angle))*(core_size)*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))',
            'sqrt(2)*cos(charge_2*(atan2(x[1]-y2,x[0]-x2)+direction_angle_2))*(core_size)*sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*sqrt((0.34+0.07*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)))*1/(1+0.41*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))+0.07*(core_size)*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))))',
            'sqrt(2)*sin(charge_2*(atan2(x[1]-y2,x[0]-x2)+direction_angle_2))*(core_size)*sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*sqrt((0.34+0.07*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)))*1/(1+0.41*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))+0.07*(core_size)*(core_size)*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))*((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2))))',
            "0.001"),x0=x0,y0=y0,x2=x2,y2=y2,direction_angle=direction_angle,direction_angle_2=direction_angle_2,core_size=1,charge=charge,charge_2=charge_2, degree=1)

        return S_int_1

    def single_one_layer_defect(self,x0=0,y0=0,x2=0,y2=0,direction_angle=3.14,charge=1,direction_angle_2=3.14/2,charge_2=1):
        S_int_1 = Expression(
            ('sqrt(2)*cos(charge*(atan2(x[1]-y0,x[0]-x0)+direction_angle))*(core_size)*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))',
            'sqrt(2)*sin(charge*(atan2(x[1]-y0,x[0]-x0)+direction_angle))*(core_size)*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))',
            '1',
            '1',
            "0.001"),x0=x0,y0=y0,x2=x2,y2=y2,direction_angle=direction_angle,direction_angle_2=direction_angle_2,core_size=1,charge=charge,charge_2=charge_2, degree=1)

        return S_int_1



    def two_defect_zero_wnt_flex(self,x0,y0,x3=0,y3=0,x_bot1=0,y_bot1=0,x_bot2=0,y_bot2=0,charge_1=1,charge_2=1,charge_3=0,charge_4=0,direction_angle_1=0,direction_angle_2=0,direction_angle_3=0,direction_angle_4=0):
        S_int_2_a = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
                ,'0.01'),A=1,z=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        S_int_2_b = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2)))*1/(1+0.41*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))+0.07*(core_size)*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2)))*1/(1+0.41*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))+0.07*(core_size)*(core_size)*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))*((x[0]-x_bot2)*(x[0]-x_bot2)+(x[1]-y_bot2)*(x[1]-y_bot2))))'\
            ,'0.01'),A=1,z=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        return S_int_2_a+S_int_2_b
        
    def two_defect_nematic_polar(self,x0,y0,x3=0,y3=0,x_bot1=0,y_bot1=0,x_bot2=0,y_bot2=0,charge_1=1,charge_2=1,charge_3=0,charge_4=0,direction_angle_1=0,direction_angle_2=0,direction_angle_3=0,direction_angle_4=0):
        S_int_2_a = Expression(\
            ('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)*cos(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
            ,'sqrt(2)*sin(charge_3*(atan2(x[1]-y_bot1,x[0]-x_bot1)+direction_angle_3)+charge_4*(atan2(x[1]-y_bot2,x[0]-x_bot2)+direction_angle_4))*sqrt((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*sqrt((0.34+0.07*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1)))*1/(1+0.41*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))+0.07*(core_size)*(core_size)*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))*((x[0]-x_bot1)*(x[0]-x_bot1)+(x[1]-y_bot1)*(x[1]-y_bot1))))'\
                ,'0.01'),A=1,z=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        S_int_2_b = Expression(\
            ('sqrt(2)*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'0'\
            ,'0'\
            ,'0.01'),A=1,z=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x_bot1=x_bot1,y_bot1=y_bot1,x_bot2=x_bot2,y_bot2=y_bot2,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        return S_int_2_a+S_int_2_b


    def two_defect_zero_wnt(self,x0,y0,x3=0,y3=0,charge_1=1,charge_2=1,charge_3=0,charge_4=0,direction_angle_1=0,direction_angle_2=0,direction_angle_3=0,direction_angle_4=0):
        S_int_2_a = Expression(('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y0,x[0]-x0)+direction_angle_3)+charge_4*(atan2(x[1]-y1,x[0]-x1)+direction_angle_4))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y0,x[0]-x0)+direction_angle_3)+charge_4*(atan2(x[1]-y1,x[0]-x1)+direction_angle_4))*sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*sqrt((0.34+0.07*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)))*1/(1+0.41*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))+0.07*(core_size)*(core_size)*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))*((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))))'\
                ,'0.01'),A=1,z=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x3=x3,y3=y3,x4=-x3,y4=y3,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        S_int_2_b = Expression(('sqrt(2)/2*cos(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*sin(charge_1*(atan2(x[1]-y0,x[0]-x0)+direction_angle_1)+charge_2*(atan2(x[1]-y1,x[0]-x1)+direction_angle_2))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*cos(charge_3*(atan2(x[1]-y0,x[0]-x0)+direction_angle_3)+charge_4*(atan2(x[1]-y1,x[0]-x1)+direction_angle_4))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'sqrt(2)/2*sin(charge_3*(atan2(x[1]-y0,x[0]-x0)+direction_angle_3)+charge_4*(atan2(x[1]-y1,x[0]-x1)+direction_angle_4))*sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*sqrt((0.34+0.07*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)))*1/(1+0.41*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))+0.07*(core_size)*(core_size)*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))*((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1))))'\
            ,'0.01'),A=1,z=1,x0=-x3,y0=y3,x1=-x0,y1=y0,x3=x3,y3=y3,x4=-x3,y4=y3,charge_1=charge_1,charge_2=charge_2,charge_3=charge_3,charge_4=charge_4,direction_angle_1=direction_angle_1,direction_angle_2=direction_angle_2,direction_angle_3=direction_angle_3,direction_angle_4=direction_angle_4,core_size=1, degree=1)

        return S_int_2_a+S_int_2_b
        






