# %% Importing
#########################################################
#########################################################

import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


# get path of this script
#path_this_script = os.getcwd()
path_this_script = os.path.realpath(sys.argv[0])

# add the ./src/ path to the search path
path_this_script_splitted = os.path.split(path_this_script)
this_script_filename = path_this_script_splitted[1]
path_this_script_splitted = os.path.split(path_this_script_splitted[0])
path_to_src = os.path.join(path_this_script_splitted[0], 'src')
print(path_to_src)
sys.path.append(path_to_src)  # I could have used sys.path.append('../src/'), but it didn't work with the debugger
path_to_cache = os.path.join(path_this_script_splitted[0], 'cache')
print(path_to_cache)
from sample_script import *

from fenics_function_space_construct import *
from Defects_3D import * 
from fenics_mesh import *
from fenics_visualization import *
#from fenics import SphereMesh
#%% 
xdmf_parent_dir=path_to_cache+'/xdmf_files1/'
sigma_small=2
sigma_large=5
Na1=1
Na2=1
height_array=np.linspace(6,10,Na1)
sigma_array=np.linspace(sigma_small,sigma_large,Na2)

file_loc=path_to_cache+'/G_height_6_sigma_2_c'#'/sigma_'+str(round(np.max(sigma_array),2))+'-'+str(round(np.min(sigma_array),2))+'_Height_'+str(round(np.max(height_array),2))+'-'+str(round(np.min(height_array),2))+'_Uniform_init_b' 
try:
    os.mkdir(file_loc)
except:
    shutil.rmtree(file_loc)
    os.mkdir(file_loc)

for j in range(0,Na2):
    wnt_strength=0 #wnt_strength_array[j]
    sigma=2#sigma_array[j]
    
    for i in range(0,Na1):
        npts = 40
        p = []
        #x_max=35

        height_fuc=height_array[i]
        print(sigma)
        print(height_fuc)
        x_max=6*sigma#max([height_fuc,sigma])+4#height_fuc+1
        #---------------------------- -------------------------------------------
        xdmf_file_name='gaussian_mesh_sigma_'+str(round(sigma,2))+'_range_'+str(round(x_max,2))+'_height_'+str(round(height_fuc,2))
        #----------------------------Boundary -------------------------------------------

        mesh_obj=Mesha()
        
        loc=xdmf_parent_dir+xdmf_file_name+'.xdmf'
        #mesh_fc=mesh_obj.load_mesh(loc)
        mesh_obj.load_mesh(loc)
        #mvc = MeshValueCollection("size_t", mesh, 1)

        z0=height_fuc*0.5
        refine_density=2
        mesh_obj.region_refine_mesh(0.05,refine_density)
        bulk_parts,top_bool,vertex_val=mesh_obj.mark_region(0.05)
        mesh_cood=mesh_obj.coordinates(type='bmesh')
        sys_size=mesh_cood.shape

        x, y,z= symbols('x[0], x[1],x[2]')
        f = height_fuc*sp.exp(-((x)**2+(y)**2)/(2*sigma**2))-z
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        dz_f=f.diff(z, 1)

        mag_d=sp.sqrt(dx_f**2+dy_f**2+dz_f**2)
        d_fE=Expression((ccode(-dx_f/mag_d).replace('M_PI', 'pi'),ccode(-dy_f/mag_d).replace('M_PI', 'pi'),ccode(-dz_f/mag_d).replace('M_PI', 'pi'),'0','0'),degree=1)
        mesh_obj.normal_expression(d_fE)



        mesh=mesh_obj.out(type='bmesh')

        mix_order=5
        P2=FiniteElement('CG',mesh.ufl_cell(),1)
        # define first function space for z 
        Func_space=Fenics_function_space(mesh,sys_size=sys_size,bulk_markers=bulk_parts,finite_element=P2,mix_order=mix_order,file_dir=file_loc)
        Df=Defects()
        x0=0
        y0=0

        #Init_val=Df.single_defect(x0=x0,y0=y0, charge_1=2,direction_angle_1=2)
        Init_val=Expression(('0','-1/3','0','-1/3','0'),degree=1)
        #Init_val=Df.single_defect(x0=0,y0=0, charge_1=2,direction_angle_1=0)
        Func_space.project_to_function_space(Init_val)
        Func_space.add_noise(0.01)
        Func_space.general_normal_projector(d_fE)
        S_order=Func_space.S_order()
        dir_density=45#+int(max([height_fuc,sigma]))
        Dir_mesh=mesh_cood[::dir_density]
        s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood)

        unit_center_angle=[0,0,0] # first term rotate around x, then, y ,then z. 
        concentration_fuc,full_dir_val,R_mat=Func_space.Gradient(sigma,height_fuc,type="gauss",alpha=1,unit_center_angle=unit_center_angle)
        Dir_profile=Func_space.Q_tensor_np(dir_density)

        view_elev=0
        view_azim=0

        azim_change=90  
        elev_chage=70

        #vs.save_fig(file_loc,0, 200)
        dt=0.2

        K=1
        const=0.5

        Func_space.variational_nematic(dt,K=K,const=const,wnt_strength=wnt_strength,Wnt_on=False)
        N=0
        t=0
        plot_density=120 
        plot_index=1
        Defect_loc_arry=kmean_center
        Defect_relative_info=np.array([[0,0,0,0,0,0,0,0,0,0,0,0]])
        check_num=0
        Target_num=2
        condition_check=0
         
        vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=x_max,row_num=1,column_num=1)
        vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True)
        vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Back',view_elev=view_elev, view_azim=view_azim,color='black',vector_length=0.13*height_fuc,vector_width=0.5)
        vs.plot_wnt_range(plot_index=1,sigma=sigma,z0=height_fuc*np.exp(-1/2),alpha=1,R=height_fuc,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev, view_azim=view_azim,sub_fig=False,alpha_p=0.5,surf=True)


        vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=x_max,row_num=1,column_num=1)
        vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev+elev_chage, view_azim=view_azim,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True)
        vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Back',view_elev=view_elev+elev_chage, view_azim=view_azim,color='black',vector_length=0.13*height_fuc,vector_width=0.5)

        vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=x_max,row_num=1,column_num=1)
        vs.plot_wnt_range(plot_index=1,sigma=sigma,z0=height_fuc*np.exp(-1/2),alpha=1,R=height_fuc,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev, view_azim=view_azim,sub_fig=False,alpha_p=0.5,surf=True)
        vs.plot_wnt_range(plot_index=1,sigma=sigma,z0=height_fuc*np.exp(-1/2),alpha=1,R=height_fuc,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev, view_azim=view_azim,sub_fig=False,alpha_p=0.5,surf=True)

        #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True)
        #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Back',view_elev=view_elev, view_azim=view_azim,color='black',vector_length=0.07*height_fuc,vector_width=1)
        #vs.plot_func_surface(field_cood=concentration_fuc, Ct_id=0,plot_index=3,view_elev=view_elev, view_azim=view_azim,title='Gaussian Concentration Strength: '+str(wnt_strength),color_range=[0,1],set_range=False)
        
        #vs.save_fig(file_loc,1, 200,name='gaussian')

        set_log_level(LogLevel.ERROR)
        while condition_check<Target_num:
            #Func_space.get_jump_value()
            t=t+dt
            #Func_space.singular_bc(R=R,beta=beta)

            print('Current step:'+str(N))
            print('Current time:'+str(round(t,2)))
            print('-----------------')
            Func_space.evolve(dt=dt,apply_bc= False)
            
            condition_check_density=2
            if (N % plot_density)==0:
                
                S_order=Func_space.S_order()
                
                s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood,0.3)
                Dir_profile=Func_space.Q_tensor_np(dir_density)
                N_error=Func_space.normal_error()
                #Defect_loc_arry=Func_space.defect_loc_arry(Defect_loc_arry,kmean_center)
                print('-----------------')
                print('Max Order Parameter: '+str(S_order.max()))
                print('Mmin Order Parameter :'+str(S_order.min()))
                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=x_max,row_num=1,column_num=1)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='',view_elev=view_elev, view_azim=view_azim+45,color='black',vector_length=0.13*height_fuc,vector_width=1)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+45,title='',color_range=[0,1],set_range=True,bg_off=True,cbar=False)
                vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_range_'+str(round(sigma,2))+'_Step_'+str(N)+'A', transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=x_max,row_num=1,column_num=1)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='',view_elev=90, view_azim=0,color='black',vector_length=0.13*height_fuc,vector_width=1)
               # vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=90, view_azim=0,title='',color_range=[0,1],set_range=True,bg_off=True,cbar=False)
                vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_range_'+str(round(sigma,2))+'_Step_'+str(N)+'B', transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=x_max,row_num=1,column_num=1)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='',view_elev=view_elev, view_azim=view_azim+45,color='black',vector_length=0.13*height_fuc,vector_width=1)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+45+180,title='',color_range=[0,1],set_range=True,bg_off=True,cbar=False)
                vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_range_'+str(round(sigma,2))+'_Step_'+str(N)+'C', transparent=True)

                #vs.plot_func_surface(field_cood=concentration_fuc, Ct_id=0,plot_index=3,view_elev=view_elev, view_azim=view_azim+45,title='Gaussian Concentration Strength: '+str(round(wnt_strength,2)),color_range=[0,1],set_range=True,bg_off=False,cbar=True)
                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=x_max,row_num=1,column_num=1)
                current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,sigma=sigma,plot_index=1,z0=height_fuc*np.exp(-1/2),alpha=1,R=height_fuc,region_max=x_max,title='',view_elev=view_elev, view_azim=view_azim+45,sub_fig=True,ring_density=10,sub_fig_on=False)
                print(current_properties_array)
                Defect_relative_info=Func_space.defect_loc_arry(Defect_relative_info,current_properties_array)
                #
                #vs.plot_func_surface(field_cood=N_error, Ct_id=0,plot_index=4,view_elev=view_elev, view_azim=view_azim+45,title='Normal Error',color_range=[0,1],set_range=False,cbar=False,alpha_c=0.1)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='',view_elev=view_elev+20, view_azim=view_azim+45,color='black',vector_length=0.13*height_fuc,vector_width=1)
                #$vs.plot_func_surface(field_cood=s_val, Ct_id=1,plot_index=1,view_elev=view_elev, view_azim=view_azim+45,title=' '+str(round(height_fuc,2))+'/'+str(round(sigma,2)),color_range=[0,1],set_range=True,cbar=False,alpha_c=0.4,bg_off=False)
                #current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,plot_index=8,z0=z0,alpha=alpha,R=R,title='Defect Position and lobal Director',view_elev=view_elev+45, view_azim=view_azim+45,sub_fig=True,ring_density=10)
                vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_range_'+str(round(sigma,2))+'_Step_'+str(N)+'D', transparent=True)
                check_val=Func_space.defect_check(Defect_relative_info[plot_index,:],Defect_relative_info[plot_index-1,:],target=0.1)
                if check_val==True:
                    check_num=check_num+1
                    print('Converge step: ' +str(check_num)+'/'+str(Target_num))
                    
                    condition_check=condition_check+1
                    print('condition_check: ' +str(condition_check)+'/'+str(condition_check_density))
                else:
                    condition_check=0

                if condition_check==condition_check_density:
                    vs.save_fig(file_loc,plot_index, 200,name='Sigma_'+str(round(sigma,2))+'_Height_'+str(round(height_fuc,2))+'_Step_'+str(N))
                else:
                    pass
                
                plot_index=plot_index+1
            if N>10000:
                condition_check=1+Target_num
                vs.save_fig(file_loc,plot_index, 200,name='Sigma_'+str(round(sigma,2))+'_Height_'+str(round(height_fuc,2))+'_Step_'+str(N))
            vs.close_fig()

            N=N+1
        print('total time to converge: '+str(round(t,2)))
        print('total step to converge: '+str(N))
        header_file=np.array([""])
