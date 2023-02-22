# %% Importing
#########################################################
#########################################################

import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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

from fenics_2D_function_space_construct import *
from Defects_3D import * 

from fenics_visualization import *
from fenics_mesh import *
#from fenics import SphereMesh
#%% 


#----------------------------Boundary -------------------------------------------
R=150
z0=0.5*R
raw_density=300
Run_n=20
Run_n_z0=10
wnt_strength_array_bare=np.linspace(0.3,10,Run_n)
#wnt_strength_array=np.linspace(340,390,Run_n)
z0_array=np.linspace(50.0,80.0,Run_n_z0)
#file_loc=path_to_cache+'/wnt_strength_'+str(np.max(wnt_strength_array))+'-'+str(np.min(wnt_strength_array))+'_Range_'+str(np.max(z0_array))+'-'+str(np.min(z0_array))+'_N_wnt'+str(Run_n)+'_N_z0'+str(Run_n_z0)+'_2p1_init_Large_Large_beta'#+'_Seed2022_Noise_level0.5_init'
file_loc=path_to_cache+'/R15-80_0.3-10_density250'
defect_csv_file_loc='../cache/data/raw_data/R22-80_0.3-10_density250.csv'
defect_csv_file_loc_bare='../cache/data/raw_data/R22-80_0.3-10_density250_bare.csv'
try:
    os.mkdir(file_loc)
except:
    shutil.rmtree(file_loc)
    os.mkdir(file_loc)

matrix_dat=np.zeros((Run_n,Run_n_z0))
pd.DataFrame(matrix_dat).to_csv(defect_csv_file_loc,mode='w+', header=None, index=None)
matrix_dat=np.zeros((Run_n,Run_n_z0))
pd.DataFrame(matrix_dat).to_csv(defect_csv_file_loc_bare,mode='w+', header=None, index=None)
for j in range(0,Run_n_z0):
    #z0=21.31#z0_array[j]#0.79*j+4.16#7.9#z0_array[j]
    z0=z0_array[j]#0.79*j+4.16#7.9#z0_array[j]

    for i in range(0,Run_n):

        #wnt_strength=-303.16+i*5.26#-(100+5.26*6*(i+1))#wnt_strength_array[i]
        wnt_strength=-z0*z0*wnt_strength_array_bare[i]#-(100+5.26*6*(i+1))#wnt_strength_array[i]
 
        mesh_obj=Mesha()
        unit_normal=mesh_obj.square(size=R,mesh_density=raw_density,refine_density=0)
        #mesh_obj.sphere_mesh(R,20,rot=False)

        
        



        mesh=mesh_obj.out()
        mesh_cood=mesh_obj.coordinates()
        sys_size=mesh_cood.shape
        Df=Defects()
        mix_order=2
        P2=FiniteElement('CG',mesh.ufl_cell(),1)
        # define first function space for z 
        Func_space=Fenics_function_space(mesh,sys_size=sys_size,finite_element=P2,mix_order=mix_order,file_dir=file_loc)
        #Init_val=Expression(('1','0'),degree=1)
        Init_val=Df.single_defect(x0=0,y0=0, charge_1=2,direction_angle_1=0)
        #Init_val=Df.two_defect (x0=12.5,y0=0, charge_1=1,charge_2=1 ,direction_angle_1=0,direction_angle_2=0 )
        Func_space.project_to_function_space(Init_val)
        Func_space.add_noise(0.01,seed=2021)
        S_order=Func_space.S_order()

        s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood)
        concentration_fuc,grad_mag,full_dir_val=Func_space.Gradient(sigma=z0,type="gauss")
        dir_density=45
        #Dir_profile=Func_space.Q_tensor_np(dir_density)
        Dir_mesh=mesh_cood[::dir_density]

        vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=2,column_num=2)
        vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True)
        #vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=2,title='e_theta basis (respect to z axis)',color='black')


        current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,plot_index=2,z0=z0,R=R,title='Defect Position and lobal Director',sub_fig=True,ring_density=10)
       # vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=2,plot_density=2,title='e_theta basis (respect to z axis)',color='black')
#

        vs.plot_wnt_range(plot_index=2,z0=z0,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',sub_fig=False)
        vs.plot_func_surface(field_cood=concentration_fuc, Ct_id=0,plot_index=3,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True)
        #vs.plot_quiver(_director=full_dir_val,Ct_id=2,plot_index=3,plot_density=3,title='e_theta basis (respect to z axis)',color='black')

        #vs.save_fig(file_loc,0, 200,name=str(wnt_strength))



        dt=0.9

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
        Target_num=2

        set_log_level(LogLevel.ERROR)
        condition_check=0

        while condition_check<Target_num:
            #Func_space.get_jump_value()
            t=t+dt
            #Func_space.singular_bc(R=R,beta=beta)

            print('Current step:'+str(N))
            print('Current time:'+str(round(t,2)))
            print('-----------------')
            print(file_loc)
            print('Do not Touch')

            Func_space.evolve(dt=dt,apply_bc= False)
            
            condition_check_density=Target_num
            if (N % plot_density)==0:
                
                S_order=Func_space.S_order()
                s_val,mesh_loc,kmean_center=Func_space.get_defect_loc(mesh_cood,0.7)
                Dir_profile=Func_space.Q_tensor_np(dir_density)
                print('-----------------')
                print('Max Order Parameter: '+str(S_order.max()))
                print('Max Order Parameter :'+str(S_order.min()))
                #print(kmean_center)

                vs=Visualization(mesh_cood,mesh_loc,Dir_mesh,R=R,row_num=2,column_num=2)
                vs.plot_func_surface(field_cood=S_order, Ct_id=0,plot_index=1,title='Nematic on Sphere: Back',color_range=[0,1],set_range=True)
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=2,title='e_theta basis (respect to z axis)',color='black')
                vs.plot_wnt_range(plot_index=1,z0=z0,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',sub_fig=False)

                current_properties_array=vs.plot_defect_center(kmean_center=kmean_center,plot_index=2,z0=z0,R=R,title='Defect Position and lobal Director',sub_fig=True,ring_density=10)
                bare_val=vs.get_bare_defect_loc()
 
                vs.plot_quiver(_director=Dir_profile,Ct_id=2,plot_index=1,plot_density=2,title='e_theta basis (respect to z axis)',color='black')


                vs.plot_wnt_range(plot_index=2,z0=z0,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',sub_fig=False)
                vs.plot_wnt_range(plot_index=3,z0=z0,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',sub_fig=False)
                vs.plot_wnt_range(plot_index=4,z0=z0,R=R,title='Wnt Strength: '+str(wnt_strength),color='red',sub_fig=False)

                vs.plot_func_surface(field_cood=concentration_fuc, Ct_id=0,plot_index=3,title='Wnt Range: '+str(round(z0,2)),color_range=[0,1],set_range=True)
                vs.plot_func_surface(field_cood=-wnt_strength*grad_mag, Ct_id=0,plot_index=4,title='Wnt Range: '+str(round(z0,2)),color_range=[0,1],set_range=False)

                
                #vs.plot_quiver(_director=full_dir_val,Ct_id=2,plot_index=3,plot_density=3,title='e_theta basis (respect to z axis)',color='black')
                Defect_relative_info=Func_space.defect_loc_arry(Defect_relative_info,current_properties_array)
                check_val=Func_space.defect_check(Defect_relative_info[plot_index,:],Defect_relative_info[plot_index-1,:],target=0.1)
                Func_space.defect_out(row_ele=Run_n,col_ele=Run_n,row_ele_index=i,col_ele_index=j,checked_array=Defect_relative_info[plot_index,:],csv_path=defect_csv_file_loc)
                Func_space.defect_bare_out(row_ele=Run_n,col_ele=Run_n,row_ele_index=i,col_ele_index=j,checked_array=bare_val,csv_path=defect_csv_file_loc_bare)

                #vs.save_fig(file_loc,plot_index, 200,name=str(wnt_strength+N))
                #vs.save_fig(file_loc,0, 200,name=str(wnt_strength+t))
                #vs.save_fig(file_loc,plot_index, 200,name=str(round(wnt_strength,2)+N))
                if check_val==True:
                    check_num=check_num+1
                    print('Converge step: ' +str(check_num)+'/'+str(Target_num))
                    print('condition_check: ' +str(condition_check)+'/'+str(condition_check_density))
                    condition_check=condition_check+1
                else:
                    condition_check=0

                if condition_check==condition_check_density:
                    vs.save_fig(file_loc,plot_index, 200,name=str(round(wnt_strength,2))+'_range_'+str(round(z0,2))+'_N_'+str(N))
                else:
                    pass
                
                plot_index=plot_index+1
            if N>10000:
                condition_check=1+Target_num
                vs.save_fig(file_loc,plot_index, 200,name=str(round(wnt_strength,2))+'_range_'+str(round(z0,2)))
            vs.close_fig()
            N=N+1
        print('total time to converge: '+str(round(t,2)))
        print('total step to converge: '+str(N))
        print('Current-Wnt-Strength: '+str(wnt_strength))
        print('Current-range: '+str(z0))
