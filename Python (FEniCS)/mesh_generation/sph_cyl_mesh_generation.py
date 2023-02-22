import sys
import meshio
import gmsh
import numpy as np
path_to_cache='../cache'
gmsh_parent_dir=path_to_cache+'/gmsh_files/'
xdmf_parent_dir=path_to_cache+'/xdmf_files/'
sigma_small=2
sigma_large=5
Na1=1
Na2=1
sigma=3 
bot_aspect_array=np.linspace(4,28,Na1)
height_array=np.linspace(5,30,Na2)
top_aspect_array=np.linspace(1,28,Na1)


for j in range(0,Na2):

    for i in range(0,Na1):
        npts = 5
        npts_top=5
        npts_bot=5
        p = []
        
        #sigma=2

        height_fuc=height_array[j]
        bot_aspect=bot_aspect_array[i]
        top_aspect=top_aspect_array[i]
        lc = 0.3
        x_max=height_fuc#1.4*max([height_fuc,sigma])+4#height_fuc+1
        x_min=-height_fuc

        #---------------------------- -------------------------------------------
        x_s=-np.linspace(-x_max,-x_min,npts)
        xdmf_file_name='sphere_cyl_mesh_sigma_'+str(round(sigma,2))+'_range_'+str(round(x_max,2))+'_height_'+str(round(height_fuc,2))+'_bot_aspect_'+str(round(bot_aspect,2))+'_top_aspect_'+str(round(top_aspect,2))
        gmsh_file_name='sphere_cyl_mesh_sigma_'+str(round(sigma,2))+'_range_'+str(round(x_max,2))+'_height_'+str(round(height_fuc,2))+'_bot_aspect_'+str(round(bot_aspect,2))+'_top_aspect_'+str(round(top_aspect,2))

        gmsh.initialize()
        gmsh.model.add(gmsh_file_name)
        gmsh.model.occ.synchronize()
        x_top_sphere=-np.linspace(-(top_aspect*sigma+x_max),-x_max,npts_top)
        print(x_top_sphere)
        x_bot_elp=np.linspace(-x_max,-(bot_aspect*sigma+x_max),npts_bot)
        print(x_bot_elp)
        c_num=len(p)
        for  i in range(0, npts_top):
            c_num=len(p)
           # print(c_num)
            x=np.sqrt(0.01+sigma**2-(x_top_sphere[i]-x_max)**2/(top_aspect**2))
            numa=gmsh.model.occ.addPoint(x,0,x_top_sphere[i], lc, 1000 + c_num)
            
            p.append(1000+ c_num)
            print(x)
            print(x_top_sphere[i])
            
        print('----')
        for i in range(1, npts):
            c_num=len(p)
            #print(p)
            #print(c_num)
            cyl=sigma#*np.exp(-(x_s[i]*x_s[i])/(2*sigma**2))

            numa=gmsh.model.occ.addPoint(sigma,0,x_s[i], lc, 1000 + c_num)
            
            p.append(1000+ c_num)
            print(sigma)
            print(x_s[i])
            #p_rm_array.append((0,1000+ i) )
        print('----')
        for  i in range(1, npts_bot):
            c_num=len(p)
           # print(c_num)
            x=np.sqrt(sigma**2-(x_bot_elp[i]-x_min)**2/(bot_aspect**2))
            numa=gmsh.model.occ.addPoint(x,0,x_bot_elp[i], lc, 1000 + c_num)
            
            p.append(1000+ c_num)
            print(x)
            print(x_bot_elp[i])
            
        



        gmsh.model.occ.addSpline(p, 1003)
        #gmsh.model.occ.addWire([1003], 1000)
       
        gmsh.model.occ.addCircle(0, 0, 0, sigma, 1001)
        gmsh.model.occ.addWire([1001], 1001)
        gmsh.model.occ.addPipe([(1, 1003)], 1001, 'DiscreteTrihedron')
        gmsh.model.occ.remove([(1,1001),(1,1003)])
        p_rm_array1=[]
        p_rm_array0=[]
        for i in range(0, 1000):
            p_rm_array0.append((0,1000+ i) )
            p_rm_array1.append((1,1000+ i) )
        gmsh.model.occ.remove(p_rm_array0)
        gmsh.model.occ.remove(p_rm_array1)
        #gmsh.model.occ.remove(p_rm_array)
        
        
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(gmsh_parent_dir+gmsh_file_name+'.msh')

        gmsh.finalize()
        
        msh = meshio.read(gmsh_parent_dir+gmsh_file_name+'.msh')
        for cell in msh.cells:
            if cell.type == "triangle":
                triangle_cells = cell.data
            elif  cell.type == "tetra":
                tetra_cells = cell.data


        triangle_mesh =meshio.Mesh(points=msh.points,
                                cells=[("triangle", triangle_cells)])
        #meshio.write("mesh.xdmf", tetra_mesh)
        loc=xdmf_parent_dir+xdmf_file_name+'.xdmf'
        meshio.write(loc, triangle_mesh)
        print('Done')
        
    #from fenics import SphereMesh
    #%% 

#----------------------------Boundary -------------------------------------------
