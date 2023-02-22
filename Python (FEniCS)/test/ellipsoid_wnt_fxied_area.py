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

Run_n_wnt=1
Run_n_z=20

wnt_strength_array=np.linspace(0,5.5,Run_n_wnt)
#z0_array=np.linspace(0.8*R,0.9*R,Run_n_z)
ratio_array=np.linspace(1,2.5,Run_n_z)

es=np.sqrt(1-1/ratio_array**2)
area_spheriod=2*np.pi*(1+ratio_array/(es+0.001)*np.arcsin(es+0.001))
R_array=np.sqrt(Area_sphere/area_spheriod)
print(ratio_array)
print(R_array)
print(R_array*ratio_array)
file_loc=path_to_cache+'/s_elip_fine_beta_N_6'#'/wnt_strength_'+str(np.round(np.max(ratio_array),2))+'-'+str(np.round(np.min(ratio_array),2))+'_Range_'+str(np.round(np.max(z0_array)))+'-'+str(np.round(np.min(z0_array)))+'_Strength_'+str(np.round(np.max(wnt_strength_array)))+'-'+str(np.round(np.min(wnt_strength_array)))
# try:
#     os.mkdir(file_loc)
# except:
#     shutil.rmtree(file_loc)
#     os.mkdir(file_loc)
z0_sphere=0.6
r0_3D=R_bare*np.sqrt((1-z0_sphere**2))
z0_upper=(1-z0_sphere)*R_bare
d_fixed=np.sqrt(r0_3D**2+z0_upper**2)
print(d_fixed)
print('sdfsagv')



for j in range(12,Run_n_z):
    
    R=R_array[j]
    

    alpha=ratio_array[j]
    raw_density=int(32*alpha)

    z0=z0_sphere*R*alpha#z0_array[j]
    #print(z0)
    for i in range(0,Run_n_wnt):

        #wnt_strength_array[i]


        #a= Area_sphere -2*np.pi*x**2-2*np.pi*np.sqrt(1-alpha**2)*
        #fsolve
        #alpha=3.4#ratio_array[i]
        mesh_obj=Mesha()
        unit_normal=mesh_obj.ellipsoid(alpha=alpha,beta=beta,R=R,mesh_density=0,raw_density=raw_density)
        #mesh_obj.sphere_mesh(R,20,rot=False)

        
        bc_tol=1E-10
        bulk_parts,top_bool,vertex_val=mesh_obj.mark_region(z0,tol=1E-10)

        mesh=mesh_obj.out(type='bmesh')
        
        # getting the mesh coorinate 
        mesh_cood=mesh_obj.coordinates(type='bmesh')
   #     mesh_cood_top=mesh_cood[top_bool]
    #    top_vertex_val=vertex_val[top_bool]
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
     #   h=0.98*R*alpha
        #Init_val=Df.single_defect(x0=x0,y0=y0, charge_1=2,direction_angle_1=2)
        #Init_val=Expression(('(sqrt(x[2]*x[2]))<h ? 1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0'),h=h,degree=1)
        #Init_val=Expression(('(sqrt(x[2]*x[2]))<h ? 0.1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0','(sqrt(x[2]*x[2]))<h ? 1: 0'),h=h,degree=1)
        #Init_val=Expression(('(sqrt(x[2]*x[2]))<h ? 0: x[0]*x[0]','(sqrt(x[2]*x[2]))<h ? 0:x[1]*x[0]','0','(sqrt(x[2]*x[2]))<h ? 0:x[1]*x[1]','0'),h=h,degree=1)
        Init_val=Expression(('-1','0','0','-1','0'),degree=1)
        Func_space.project_to_function_space(Init_val)
        Func_space.add_noise(0,seed=2021)
        Func_space.normal_projector(alpha=alpha,beta=beta,R=R)
        S_order=Func_space.S_order()
        dir_density=15
        Dir_mesh=mesh_cood[::dir_density]
        #sigma,d_z=Func_space.get_r0_loc(mesh_cood=mesh_cood,d_fixed=d_fixed,alpha=alpha,R=R)
 
      #  z0=d_z

        s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood)

        unit_center_angle=[0,0,0] # first term rotate around x, then, y ,then z. 
        #sigma=R*np.sin(np.arccos(z0/(R)))
        sigma=6
        concentration_fuc,full_dir_val=Func_space.Gradient_new(sigma,R,alpha=alpha)
        Dir_profile=Func_space.Q_tensor_np(dir_density)

        view_azim=0
        view_elev=0
        azim_change=90  
        elev_chage=90
 
        #vs.save_fig(file_loc,0, 200)
        dt=0.2

        K=1
        const=0.5
        wnt_strength=-30*sigma**2
        Func_space.variational_nematic(dt,K=K,const=const,wnt_strength=wnt_strength,Wnt_on=False)
        N=0
        t=0
        plot_density=160
        plot_index=1
        Defect_loc_arry=kmean_center
        Defect_relative_info=np.array([[0,0,0,0,0,0,0,0,0,0,0,0]])
        check_num=0
        Target_num=3  
        condition_check=0

        set_log_level(LogLevel.ERROR)
        while condition_check<Target_num:
            #Func_space.get_jump_value()
            t=t+dt
            #Func_space.singular_bc(R=R,beta=beta)

            print('Current step:'+str(N))
            print('Current time:'+str(round(t,2)))
            print('-----------------')
            Func_space.evolve(dt=dt,apply_bc= False)
            print(path_this_script)
            print('-----------------')
            print(file_loc)

            
            condition_check_density=Target_num
            if (N % plot_density)==0:
                
                S_order=Func_space.S_order()
                
                s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood,0.7)
                Dir_profile=Func_space.Q_tensor_np(dir_density)
                N_error=Func_space.normal_error()
                #Func_space.average_angle(Dir_mesh,Dir_profile,band_wdith=0.5)
                #Defect_loc_arry=Func_space.defect_loc_arry(Defect_loc_arry,kmean_center)
                print('-----------------')
                print('Max Order Parameter: '+str(S_order.max()))
                print('Max Order Parameter :'+str(S_order.min()))
                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=4,column_num=2)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=2,view_elev=view_elev, view_azim=view_azim,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True)
                #vs.plot_wnt_range(plot_index=2,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev, view_azim=view_azim,sub_fig=False,alpha_p=0.5)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=2,plot_density=1,title='Nematic on Sphere: Back',view_elev=view_elev, view_azim=view_azim,color='black',vector_length=0.07*R,vector_width=1)

                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,view_elev=view_elev, view_azim=view_azim+azim_change,title='Nematic on Sphere: Front',color_range=[0,1],set_range=True)
                #vs.plot_wnt_range(plot_index=1,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev, view_azim=view_azim+azim_change,sub_fig=False,alpha_p=0.5)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=1,title='Nematic on Sphere: Front',view_elev=view_elev, view_azim=view_azim+azim_change,color='black',vector_length=0.07*R,vector_width=1)

                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=3,view_elev=view_elev+elev_chage, view_azim=view_azim,title='Nematic on Sphere: Top',color_range=[0,1],set_range=True)
                #vs.plot_wnt_range(plot_index=3,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev+elev_chage, view_azim=view_azim,sub_fig=False,alpha_p=0.5)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=3,plot_density=1,title='Nematic on Sphere: Top',view_elev=view_elev+elev_chage, view_azim=view_azim,color='black',vector_length=0.07*R,vector_width=1)

                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=4,view_elev=view_elev-elev_chage, view_azim=view_azim,title='Nematic on Sphere: Bot',color_range=[0,1],set_range=True)
            # vs.plot_wnt_range(plot_index=4,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev-elev_chage, view_azim=view_azim,sub_fig=False,alpha_p=0.5)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=4,plot_density=1,title='Nematic on Sphere: Bot',view_elev=view_elev-elev_chage, view_azim=view_azim+azim_change,color='black',vector_length=0.07*R,vector_width=1)

                vs.plot_func_surface(field_cood=concentration_fuc, Ct_id=0,plot_index=5,view_elev=0, view_azim=view_azim+45,title='Gaussian Concentration Strength: '+str(wnt_strength),color_range=[0,1],set_range=False)
                #vs.plot_quiver(_director=full_dir_val,Ct_id=0,plot_index=5,plot_density=20,title='Top Layer Nematic Field',view_elev=view_elev+50, view_azim=view_azim+45,color='black',vector_length=0.07*R,vector_width=1)
               # vs.plot_wnt_range(plot_index=5,z0=z0,alpha=alpha,R=R,title='Strength: '+str(wnt_strength)+'  z0: '+str(round(z0,2)),color='white',view_elev=view_elev+60, view_azim=view_azim+45,sub_fig=True,alpha_p=1)

                #vs.plot_wnt_range(plot_index=6,z0=z0,alpha=alpha,R=R,title='Wnt Strength: '+str(wnt_strength),color='white',view_elev=view_elev+30, view_azim=view_azim,sub_fig=True)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=6,plot_density=1,title='e_theta basis (respect to z axis)',view_elev=view_elev+10, view_azim=view_azim+45,color='black',vector_length=0.07*R,vector_width=1)
                vs.plot_func_surface(field_cood=s_val, Ct_id=1,plot_index=6,view_elev=view_elev+10, view_azim=view_azim+45,title='Concentration Field Strength: '+str(wnt_strength),color_range=[0,1],set_range=True,cbar=False,alpha_c=0.4)
                current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,plot_index=6,z0=z0,alpha=alpha,R=R,title='Defect Position and local Director',view_elev=view_elev+10, view_azim=view_azim+45,sub_fig=True,ring_density=10)
                Defect_relative_info=Func_space.defect_loc_arry(Defect_relative_info,current_properties_array)
                #
                vs.plot_func_surface(field_cood=N_error, Ct_id=0,plot_index=7,view_elev=view_elev+45, view_azim=view_azim+45,title='Normal Error',color_range=[0,1],set_range=False,cbar=True,alpha_c=1)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=8,plot_density=1,title='e_theta basis (respect to z axis)',view_elev=view_elev+45, view_azim=view_azim+45,color='black',vector_length=0.07*R,vector_width=1)
                vs.plot_func_surface(field_cood=s_val, Ct_id=1,plot_index=8,view_elev=view_elev+45, view_azim=view_azim+45,title='Concentration Field Strength: '+str(wnt_strength),color_range=[0,1],set_range=True,cbar=False,alpha_c=0.4)
                current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,plot_index=8,z0=z0,alpha=alpha,R=R,title='Defect Position and lobal Director',view_elev=view_elev+45, view_azim=view_azim+45,sub_fig=True,ring_density=10)
               # vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_alpha_'+str(round(alpha,2))+'_wnt_range_'+str(round(z0,2)))

                check_val=Func_space.defect_check(Defect_relative_info[plot_index,:],Defect_relative_info[plot_index-1,:],target=0.1)
                if check_val==True:
                    check_num=check_num+1
                    print('Converge step: ' +str(check_num)+'/'+str(Target_num))
                    print('condition_check: ' +str(condition_check)+'/'+str(condition_check_density))
                    condition_check=condition_check+1
                else:
                    condition_check=0

                if condition_check==condition_check_density:
                    index_arry=np.array([sigma,wnt_strength])
                    Func_space.save(array=kmean_center,index_arry=index_arry,append_=True,data_file=file_loc+'/data.csv')
                    vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_alpha_'+str(round(alpha,2))+'_wnt_range_'+str(round(sigma,2))+'_z_range_'+str(round(z0,2)))
                else:
                    pass
                
                plot_index=plot_index+1
            if N>100000:
                check_num=1+Target_num
                vs.save_fig(file_loc,plot_index, 200,name='Strength_'+str(round(wnt_strength,2))+'_alpha_'+str(round(alpha,2))+'_wnt_range_'+str(round(sigma,2))+'_z_range_'+str(round(z0,2)))
            vs.close_fig()
            N=N+1
        print('total time to converge: '+str(round(t,2)))
        print('total step to converge: '+str(N))
        header_file=np.array([""])
    #np.savetxt(path_to_cache+"/defect_position.csv", Defect_loc_arry, delimiter=",")
    #np.savetxt(path_to_cache+"/defect_properties.csv", Defect_relative_info, delimiter=",")
