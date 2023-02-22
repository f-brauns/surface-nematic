import sys
import meshio
import gmsh
import numpy as np
path_to_cache='../cache'
gmsh_parent_dir=path_to_cache+'/gmsh_files1/'
xdmf_parent_dir=path_to_cache+'/xdmf_files1/'
sigma_small=2
sigma_large=5
Na1=20
Na2=10
height_array=np.linspace(15.5,28,Na1)
sigma_array=np.linspace(sigma_small,sigma_large,Na2)
for j in range(0,Na2):
    sigma=sigma_array[j]

    for i in range(0,Na1):
        npts = 40
        p = []
        
        #sigma=2

        height_fuc=height_array[i]
        lc = 0.4
        x_max=6*sigma#1.4*max([height_fuc,sigma])+4#height_fuc+1

        #---------------------------- -------------------------------------------
        x_s=np.linspace(0,x_max,npts)
        xdmf_file_name='gaussian_mesh_sigma_'+str(round(sigma,2))+'_range_'+str(round(x_max,2))+'_height_'+str(round(height_fuc,2))
        gmsh_file_name='gaussian_mesh_sigma_'+str(round(sigma,2))+'_range_'+str(round(x_max,2))+'_height_'+str(round(height_fuc,2))

        gmsh.initialize()
        gmsh.model.add(gmsh_file_name)
        gmsh.model.occ.synchronize()

        for i in range(0, npts):

            exp_fuc=height_fuc*np.exp(-(x_s[i]*x_s[i])/(2*sigma**2))
            numa=gmsh.model.occ.addPoint(x_s[i],exp_fuc,0, lc, 1000 + i)
            p.append(1000+ i)
            #p_rm_array.append((0,1000+ i) )


        gmsh.model.occ.addSpline(p, 1003)
        gmsh.model.occ.addWire([1003], 1000)
        gmsh.model.occ.rotate([(1, 1003)], 0, 0, 0, 1, 0, 0, np.pi / 2)
        gmsh.model.occ.addCircle(0, 0, 0, x_max, 1001)
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
