# %% Importing
#########################################################
#########################################################

from sample_script import *



class Mesha:
    ''' Process the gmesh file into data array, and export into a .dat format'''
    def __init__(self,file_dir='/home/zhwang/hydra_fenics/nematic_project/cache/nematic_fig_2cp1'):
        pass 
    def load_mesh(self,loc):
        '''loading mesh from the file'''
        #self.mesh = Mesh (loc)
        self.bmesh = Mesh()
        with XDMFFile(loc) as infile:
            infile.read(self.bmesh)


    def ellipsoid(self,alpha,beta,R,mesh_density,raw_density=20):
        self.alpha=alpha
        self.beta=beta
        self.R=R
        center = Point()
        ellipsoid_mesh=Ellipsoid( center,beta*R,beta*R,alpha*R)
        mesh = generate_mesh(ellipsoid_mesh, raw_density)
        self.mesh=mesh
        self.bmesh = BoundaryMesh(self.mesh, "exterior")
       # print(self.bmesh.topology().dim())
        for i in range(0,mesh_density):
            self.bmesh=refine(self.bmesh)
        global_normal = Expression((" x[0]/(a*a)", " x[1]/(b*b)", " x[2]/(c*c)"),a=beta*R,b=beta*R,c=alpha*R, degree=1)
        self.bmesh.init_cell_orientations(global_normal)
        
    def cylinder(self,R,height,mesh_density,raw_density=20):
        self.R=R
        top_center = Point(0,0,height)
        bot_center= Point(0,0,-height)
        cylinder_mesh=Cylinder( bot_center,top_center,R,R)
        mesh = generate_mesh(cylinder_mesh, raw_density)
        self.mesh=mesh
        self.bmesh = BoundaryMesh(self.mesh, "exterior")
       # print(self.bmesh.topology().dim())
        for i in range(0,mesh_density):
            self.bmesh=refine(self.bmesh)
        global_normal = Expression((" x[0]/R", " x[1]/R", " 0"),R=R, degree=1)
        self.bmesh.init_cell_orientations(global_normal)
 


    def coordinates(self,type='mesh'):
        if type=='bmesh':
            return self.bmesh.coordinates()
        else:
            return self.mesh.coordinates()

    def out(self,type='mesh'):
        if type=='bmesh':
            return self.bmesh 
        else:
            return self.mesh 

    def mark_region(self,z0,tol=1E-10):
        class Top_region(SubDomain):
            def inside(self,x,on_boundary):
                return (x[2]>=z0+tol )# or((near(x[2],-z_loc,bc_tol) or near(x[2],z_loc,bc_tol)  ))

        bulk_region=Top_region()
        # define the boundary 

        bulk_parts =MeshFunction('size_t', self.bmesh, self.bmesh.topology().dim(), 0) #CellFunction("size_t", self.bmesh)
        bulk_parts.set_all(0)
        bulk_region.mark(bulk_parts,1)




        V = FunctionSpace(self.bmesh, 'CG', 1)
        bc1 = DirichletBC(V, 1, bulk_region)
        

        u = Function(V)
        
        #v=Function(V)
        bc1.apply(u.vector())
        u_val=u.compute_vertex_values()
        bc_bool=(u_val==1)
 
        

        return bulk_parts, bc_bool,u_val
    def region_refine_mesh(self,z0,refine_density,tol=1E-10):
        class Top_region(SubDomain):
            def inside(self,x,on_boundary):
                return (x[2]>=z0+tol )# or((near(x[2],-z_loc,bc_tol) or near(x[2],z_loc,bc_tol)  ))
        bulk_region=Top_region()
        self.cell_markers = MeshFunction("bool", self.bmesh, self.bmesh.topology().dim())
        self.cell_markers.set_all(False)
        bulk_region.mark(self.cell_markers,True)

        for i in range(0,refine_density):
            self.bmesh = refine(self.bmesh, self.cell_markers)
        #global_normal = Expression((" x[0]/(a*a)", " x[1]/(b*b)", " x[2]/(c*c)"), degree=1)
        #self.bmesh.init_cell_orientations(global_normal)

    def normal_expression(self,f_expresion):
        self.bmesh.init_cell_orientations(f_expresion)