
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
import cv2
import paramiko
import base64
#import mshr
#import pygmsh  
import fenics
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
#sfrom sympy.utilities.codegen import ccode
from sympy import symbols
import sympy as sp
from sympy.printing import ccode
from sklearn.cluster import KMeans
import gmsh
import pygmsh
import meshio
from PIL import Image, ImageFilter, ImageEnhance
from scipy.optimize import fsolve
import pandas as pd
from ufl import Index
