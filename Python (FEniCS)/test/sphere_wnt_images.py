# %% Importing
#########################################################
#########################################################

import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from ufl import conditional


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

#----------------------------Boundary -------------------------------------------

R_bare=6
Area_sphere=4*np.pi*R_bare**2


beta=1
raw_density=74
Run_n_wnt=1
Run_n_z=1

wnt_strength_array=np.linspace(1,7500,Run_n_wnt)
#z0_array=np.linspace(0.8*R,0.9*R,Run_n_z)
ratio_array=np.linspace(1,2.5,Run_n_z)

es=np.sqrt(1-1/ratio_array**2)
area_spheriod=2*np.pi*(1+ratio_array/(es+0.001)*np.arcsin(es+0.001))
R_array=np.sqrt(Area_sphere/area_spheriod)
print(R_array)

file_loc=path_to_cache+'/A_Sph_wnt1'#+'/wnt_strength_'+str(np.round(np.max(wnt_strength_array)))+'-'+str(np.round(np.min(wnt_strength_array)))+'_Range_'+str(np.round(np.max(z0_array)))+'-'+str(np.round(np.min(z0_array)))

try:
    os.mkdir(file_loc)
    os.mkdir(file_loc+'/a')
    os.mkdir(file_loc+'/b')
    os.mkdir(file_loc+'/c')
    os.mkdir(file_loc+'/d')
    os.mkdir(file_loc+'/e')
    os.mkdir(file_loc+'/i')
    os.mkdir(file_loc+'/g')
    os.mkdir(file_loc+'/h')
    os.mkdir(file_loc+'/j')
    os.mkdir(file_loc+'/k')

    os.mkdir(file_loc+'/a1')
    os.mkdir(file_loc+'/b1')
    os.mkdir(file_loc+'/c1')
    os.mkdir(file_loc+'/d1')
    os.mkdir(file_loc+'/i1')
    os.mkdir(file_loc+'/j1')
    os.mkdir(file_loc+'/k1')
    os.mkdir(file_loc+'/i2')
    os.mkdir(file_loc+'/net')


except:
    shutil.rmtree(file_loc)
    os.mkdir(file_loc)
    os.mkdir(file_loc+'/a')
    os.mkdir(file_loc+'/b')
    os.mkdir(file_loc+'/c')
    os.mkdir(file_loc+'/d')
    os.mkdir(file_loc+'/e')
    os.mkdir(file_loc+'/i')
    os.mkdir(file_loc+'/g')
    os.mkdir(file_loc+'/h')
    os.mkdir(file_loc+'/j')
    os.mkdir(file_loc+'/k')

    os.mkdir(file_loc+'/a1')
    os.mkdir(file_loc+'/b1')
    os.mkdir(file_loc+'/c1')
    os.mkdir(file_loc+'/d1')
    os.mkdir(file_loc+'/i1')
    os.mkdir(file_loc+'/i2')
    os.mkdir(file_loc+'/j1')
    os.mkdir(file_loc+'/k1')
    os.mkdir(file_loc+'/net')


for j in range(0,Run_n_z):
    R=R_array[j]
    
    alpha=ratio_array[j]
    z0=0.95*R*alpha#z0_array[j]
    z2=0.95*R*alpha
    print(z0)
 
    for i in range(0,Run_n_wnt):

        wnt_strength=-wnt_strength_array[i]


        mesh_obj=Mesha()
        unit_normal=mesh_obj.ellipsoid(alpha=alpha,beta=beta,R=R,mesh_density=0,raw_density=raw_density)
        #mesh_obj.sphere_mesh(R,20,rot=False)

        
        bc_tol=1E-10
        bulk_parts,top_bool,vertex_val=mesh_obj.mark_region(z0,tol=1E-10)

        mesh=mesh_obj.out(type='bmesh')

        # getting the mesh coorinate 
        mesh_cood=mesh_obj.coordinates(type='bmesh')
        mesh_cood_top=mesh_cood[top_bool]
        top_vertex_val=vertex_val[top_bool]
        sys_size=mesh_cood.shape

        #print(middle_bulk_bool.shape)

        print(sys_size)


        #----------------------------Boundary -------------------------------------------
        mix_order=5
        P2=FiniteElement('CG',mesh.ufl_cell(),1)
        # define first function space for z 
        Func_space=Fenics_function_space(mesh,sys_size=sys_size,bulk_markers=bulk_parts,finite_element=P2,mix_order=mix_order,file_dir=file_loc)
        Df=Defects()
        x0=0
        y0=0

        #Init_val=Df.single_defect(x0=x0,y0=y0, charge_1=2,direction_angle_1=2)
        Init_val=Expression(('0','0','0','0','0'),degree=1)
        Func_space.project_to_function_space(Init_val)
        Func_space.add_noise(0.01,seed=2021)
        Func_space.normal_projector(alpha=alpha,beta=beta,R=R)
        S_order=Func_space.S_order()
        dir_density=15
        Dir_mesh=mesh_cood[::dir_density]
        s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood)

        unit_center_angle=[0,0,0] # first term rotate around x, then, y ,then z. 
        unit_center_angle2=[0,np.pi/2,0]
        sigma=R*np.sin(np.arccos(z0/(R)))
        sigma=R*np.sqrt((1-z0**2/(R**2*alpha**2)))
        concentration_fuc,full_dir_val=Func_space.Gradient_new(sigma,R,alpha=alpha)
        print('hello')
        print(sigma)
        wnt_strength=wnt_strength*sigma*sigma
        Dir_profile=Func_space.Q_tensor_np(dir_density)

        view_azim=0
        view_elev=0
        azim_change=90  

        elev_chage=90

        #vs.save_fig(file_loc,0, 200)
        dt=0.3

        K=1
        const=0.5

        Func_space.variational_nematic(dt,K=K,const=const,wnt_strength=wnt_strength,Wnt_on=True)
        N=0
        t=0
        plot_density=160
        plot_index=1
        Defect_loc_arry=kmean_center
        Defect_relative_info=np.array([[0,0,0,0,0,0,0,0,0,0,0,0]])
        check_num=0
        Target_num=3
        condition_check=0
        index_num=0
        set_log_level(LogLevel.ERROR)
        marker_sizea=43
        while condition_check<Target_num:
            #Func_space.get_jump_value()
            t=t+dt

            #Func_space.singular_bc(R=R,beta=beta)

            print('Current step:'+str(N))
            print('Current time:'+str(round(t,2)))
            print('-----------------')
            Func_space.evolve(dt=dt,apply_bc= False)
            print('Current Beta'+str(wnt_strength))
            print('Current Range'+str(z0))
            print('Current Ratio'+str(alpha))
            print('------------------')  
            print(file_loc)
            print('Do not Touch this script')
            print(path_this_script)

            condition_check_density=Target_num
            if (N % plot_density)==0:
                
                S_order=Func_space.S_order()
                len_size=0.15
                s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood,0.3)
                Dir_profile=Func_space.Q_tensor_np(dir_density)
                N_error=Func_space.normal_error()
                #Defect_loc_arry=Func_space.defect_loc_arry(Defect_loc_arry,kmean_center)
                print('-----------------')
                print('Max Order Parameter: '+str(S_order.max()))
                print('Max Order Parameter :'+str(S_order.min()))
                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev+10, view_azim=view_azim,sub_fig=False,alpha_p=0.5)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Back',view_elev=view_elev, view_azim=view_azim,color='black',vector_length=0.07*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev+10, view_azim=view_azim,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.8,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/a',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
             #   vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim,sub_fig=False,alpha_p=0.2)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Back',view_elev=view_elev+10, view_azim=view_azim,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.4)
                vs.save_fig(file_loc+'/a1',plot_index, 500,name=str(index_num), transparent=True)
                vs.overlay_img(top_img_path=file_loc+'/a1/'+str(index_num)+'.png',bot_img_path=file_loc+'/a/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_a.png',alpha=1, beta=1, gamma=0)


                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
               # vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+azim_change,sub_fig=False,alpha_p=0.9)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Front',view_elev=view_elev, view_azim=view_azim+azim_change,color='black',vector_length=0.07*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+azim_change,title='Nematic on Sphere: Front',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.8,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/b',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
              #  vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+azim_change,sub_fig=False,alpha_p=0.2)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Front',view_elev=view_elev, view_azim=view_azim+azim_change,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+azim_change,title='Nematic on Sphere: Front',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.4)
                vs.save_fig(file_loc+'/b1',plot_index, 500,name=str(index_num), transparent=True)
                vs.overlay_img(top_img_path=file_loc+'/b1/'+str(index_num)+'.png',bot_img_path=file_loc+'/b/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_b.png',alpha=1, beta=1, gamma=0)


                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
              #  vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev+elev_chage, view_azim=view_azim,sub_fig=False,alpha_p=0.2)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Top',view_elev=view_elev+elev_chage, view_azim=view_azim,color='black',vector_length=0.07*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev+elev_chage, view_azim=view_azim,title='Nematic on Sphere: Top',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.8,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/c',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
               # vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev+elev_chage, view_azim=view_azim,sub_fig=False,alpha_p=0.9)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Top',view_elev=view_elev+elev_chage, view_azim=view_azim,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev+elev_chage, view_azim=view_azim,title='Nematic on Sphere: Top',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.4)
                vs.save_fig(file_loc+'/c1',plot_index, 500,name=str(index_num), transparent=True)
                vs.overlay_img(top_img_path=file_loc+'/c1/'+str(index_num)+'.png',bot_img_path=file_loc+'/c/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_c.png',alpha=1, beta=1, gamma=0)


                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                #vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev-elev_chage, view_azim=view_azim,sub_fig=False,alpha_p=0.9)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Bot',view_elev=view_elev-elev_chage, view_azim=view_azim+azim_change,color='black',vector_length=0.07*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev-elev_chage, view_azim=view_azim,title='Nematic on Sphere: Bot',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.8,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/d',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                #vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev-elev_chage, view_azim=view_azim,sub_fig=False,alpha_p=0.2)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Bot',view_elev=view_elev-elev_chage, view_azim=view_azim,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev-elev_chage, view_azim=view_azim,title='Nematic on Sphere: Bot',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.4)
                vs.save_fig(file_loc+'/d1',plot_index, 500,name=str(index_num), transparent=True)


                vs.overlay_img(top_img_path=file_loc+'/d1/'+str(index_num)+'.png',bot_img_path=file_loc+'/d/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_d.png',alpha=1, beta=1, gamma=0)


                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
               # vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+azim_change,sub_fig=False,alpha_p=0.9)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Front',view_elev=view_elev, view_azim=view_azim+azim_change,color='black',vector_length=0.07*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+0.5*azim_change,title='Nematic on Sphere: Front',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.8,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/j',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
              #  vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+azim_change,sub_fig=False,alpha_p=0.2)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Front',view_elev=view_elev, view_azim=view_azim+0.5*azim_change,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+azim_change,title='Nematic on Sphere: Front',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.4)
                vs.save_fig(file_loc+'/j1',plot_index, 500,name=str(index_num), transparent=True)
                vs.overlay_img(top_img_path=file_loc+'/j1/'+str(index_num)+'.png',bot_img_path=file_loc+'/j/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_j.png',alpha=1, beta=1, gamma=0)




                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
               # vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+azim_change,sub_fig=False,alpha_p=0.9)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Front',view_elev=view_elev, view_azim=view_azim+azim_change,color='black',vector_length=0.07*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+1.5*azim_change,title='Nematic on Sphere: Front',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.8,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/k',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
              #  vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+azim_change,sub_fig=False,alpha_p=0.2)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Front',view_elev=view_elev, view_azim=view_azim+1.5*azim_change,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+azim_change,title='Nematic on Sphere: Front',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.4)
                vs.save_fig(file_loc+'/k1',plot_index, 500,name=str(index_num), transparent=True)
                vs.overlay_img(top_img_path=file_loc+'/k1/'+str(index_num)+'.png',bot_img_path=file_loc+'/k/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_k.png',alpha=1, beta=1, gamma=0)









                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                #vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+45,sub_fig=False,alpha_p=0.9)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Bot',view_elev=view_elev, view_azim=view_azim+45,color='black',vector_length=len_size*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=s_val, Ct_id=1,plot_index=1,view_elev=view_elev, view_azim=view_azim+45,title='Nematic on Sphere: Bot',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.6,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/i',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                #vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev, view_azim=view_azim+45,sub_fig=False,alpha_p=0.9)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Bot',view_elev=view_elev, view_azim=view_azim+45,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                #vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev-elev_chage, view_azim=view_azim,title='Nematic on Sphere: Bot',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.4)
                vs.save_fig(file_loc+'/i1',plot_index, 500,name=str(index_num), transparent=True)


                vs.overlay_img(top_img_path=file_loc+'/i1/'+str(index_num)+'.png',bot_img_path=file_loc+'/i/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_i.png',alpha=1, beta=1, gamma=0)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
               # vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Bot',view_elev=view_elev, view_azim=view_azim+45,color='black',vector_length=len_size*R,vector_width=1.5)
                vs.plot_func_surface(field_cood=concentration_fuc, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+45,title='Nematic on Sphere: Bot',color_range=[0,1],set_range=True,bg_off=True,cbar=False,alpha_c=0.6,marker_size=marker_sizea,cmap='GnBu')
                vs.save_fig(file_loc+'/i2',plot_index, 500,name=str(index_num), transparent=True)

                vs.overlay_img(top_img_path=file_loc+'/net/'+str(index_num)+'_i.png',bot_img_path=file_loc+'/i2/'+str(index_num)+'.png',out_path=file_loc+'/net/'+str(index_num)+'_i_c.png',alpha=1, beta=1, gamma=0)



                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                vs.plot_quiver(_director=full_dir_val,Ct_id=0,plot_index=1,plot_density=20,title='Top Layer Nematic Field',view_elev=view_elev+60, view_azim=view_azim+45,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Strength: '+str(wnt_strength)+'  z0: '+str(round(z0,2)),color='red',view_elev=view_elev+60, view_azim=view_azim+45,sub_fig=False,alpha_p=1)
                vs.plot_func_surface(field_cood=concentration_fuc, Ct_id=0,plot_index=1,view_elev=view_elev+60, view_azim=view_azim+45,title='Gaussian Concentration Strength: '+str(wnt_strength),color_range=[0,1],set_range=False,bg_off=True,cbar=False,alpha_c=0.4,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/e',plot_index, 500,name=str(index_num), transparent=True)
                
                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                #vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',view_elev=view_elev+10, view_azim=view_azim,sub_fig=False)
                #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='e_theta basis (respect to z axis)',view_elev=view_elev+10, view_azim=view_azim+45,color='black',vector_length=0.07*R,vector_width=1.5)
                current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,plot_index=1,z0=z0,alpha=alpha,R=R,title='Defect Position and local Director',view_elev=view_elev+10, view_azim=view_azim+45,sub_fig=True,ring_density=10,sub_on=False)
                Defect_relative_info=Func_space.defect_loc_arry(Defect_relative_info,current_properties_array)
               # vs.plot_func_surface(field_cood=s_val, Ct_id=1,plot_index=1,view_elev=view_elev+10, view_azim=view_azim+45,title='Concentration Field Strength: '+str(wnt_strength),color_range=[0,1],set_range=True,cbar=False,alpha_c=0.4,bg_off=False)
                #vs.save_fig(file_loc+'/f',plot_index, 200,name=str(index_num), transparent=True)
                
                #
                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='e_theta basis (respect to z axis)',view_elev=0, view_azim=view_azim+45,color='black',vector_length=len_size*R*alpha,vector_width=1.5)
                current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,plot_index=1,z0=z0,alpha=alpha,R=R,title='Defect Position and lobal Director',view_elev=0, view_azim=view_azim+45,sub_fig=True,ring_density=10,sub_on=False)
                vs.plot_func_surface(field_cood=s_val, Ct_id=1,plot_index=1,view_elev=0, view_azim=view_azim+45,title='Concentration Field Strength: '+str(wnt_strength),color_range=[0,1],set_range=True,cbar=False,alpha_c=0.6,bg_off=False,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/g',plot_index, 500,name=str(index_num), transparent=True)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=1,column_num=1,w=10,h=10)

                vs.plot_func_surface(field_cood=N_error, Ct_id=0,plot_index=1,view_elev=view_elev+45, view_azim=view_azim+45,title='Normal Error',color_range=[0,1],set_range=False,cbar=True,alpha_c=1,bg_off=True,marker_size=marker_sizea)
                vs.save_fig(file_loc+'/h',plot_index, 500,name=str(index_num), transparent=True)



                index_num=index_num+1
                check_val=Func_space.defect_check(Defect_relative_info[plot_index,:],Defect_relative_info[plot_index-1,:],target=0.1)
                if check_val==True:
                    check_num=check_num+1
                    print('Converge step: ' +str(check_num)+'/'+str(Target_num))
                    print('condition_check: ' +str(condition_check)+'/'+str(condition_check_density))
                    condition_check=condition_check+1
                else:
                    condition_check=0

                if condition_check==condition_check_density:
                    vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_range_'+str(round(z0,2)))
                else:
                    pass
                
                plot_index=plot_index+1
            if N>100000:
                check_num=1+Target_num
                vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_range_'+str(round(z0,2)))
            vs.close_fig()
            N=N+1
        print('total time to converge: '+str(round(t,2)))
        print('total step to converge: '+str(N))
        header_file=np.array([""])
    #np.savetxt(path_to_cache+"/defect_position.csv", Defect_loc_arry, delimiter=",")
    #np.savetxt(path_to_cache+"/defect_properties.csv", Defect_relative_info, delimiter=",")
