# %% Importing
#########################################################
#########################################################

from audioop import avgpp
from math import atan2
from sample_script import *
from ufl import atan_2
class SymTransZ(Expression):
    'Given u: (x, y) --> R create v: (x, y, z) --> R, v(x, y, z) = u(x, y).'
    def __init__(self, u):
        self.u = u





class Fenics_function_space:
    ''' Process the gmesh file into data array, and export into a .dat format'''
    def __init__(self,mesh,sys_size,finite_element,bulk_markers=0,mix_order=1,file_dir='/home/zhwang/hydra_fenics/nematic_project/cache/nematic_fig_2cp2'):
        self.mesh=mesh
        self.sys_size=sys_size
        self.x, self.y, self.z = SpatialCoordinate(mesh)
        self.mix_order=mix_order
        self.n = FacetNormal(self.mesh )
        
        try: 
            self.P2=finite_element
            print("using input P2")
           # self.P_full = RestrictedElement(self.P2, dc)
            
        except:
            self.P2=FiniteElement('CG',triangle,1)
            print("using defult P2")
           # self.P_full = RestrictedElement(self.P2, dc)
        if mix_order==1:
            element=finite_element
        else:
            element_list=[self.P2] * mix_order
            element=MixedElement(element_list)
            print('mixing')
        # for the singular bc 
        self.Scalar_space=FunctionSpace(self.mesh,self.P2)
        #self.T =TensorFunctionSpace(self.mesh, 'CG', 1, shape=(3, 3))  

        self.V=FunctionSpace(self.mesh,element)
        degree = self.V.ufl_element().degree()
        #print(degree)
        # Define trial and test functions. 
        self.func = Function(self.V)
        self.test_func=TestFunction(self.V)

        q_degree=2
        dx=Measure("dx",domain=mesh)
        dx=Measure("dx",domain=mesh,subdomain_data=bulk_markers)
        dx=dx(metadata={'quadrature_degree': q_degree})
        self.dx=dx
 
        print(os.getcwd())
        # stepping 
        t = float(0)
        iii=0 # stepper 
        files=[] # file list 


    """   
    def Q_tensor(self,func):
         return func[0]*as_matrix(((func[1]*func[1],func[1]*func[2],func[1]*func[3]),\
                        (func[2]*func[1],func[2]*func[2],func[2]*func[3]), \
                        (func[3]*func[1],func[3]*func[2],func[3]*func[3])) )-0.5*(1-func[0])*self.Normal_project-(1/3)*Identity(3)

    def Q_tensor_test(self,func):
         return as_matrix(((func[0]+func[1]+func[2]+func[3],func[0]+func[1]+func[2]+func[3],func[0]+func[1]+func[2]+func[3]),\
                        (func[0]+func[1]+func[2]+func[3],func[0]+func[1]+func[2]+func[3],func[0]+func[1]+func[2]+func[3]), \
                        (func[0]+func[1]+func[2]+func[3],func[0]+func[1]+func[2]+func[3],func[0]+func[1]+func[2]+func[3])) )#-0.5*self.Normal_project-(1/3)*Identity(3)
    """



    def Q_tensor(self,func):
         return as_matrix(((func[0],func[1],func[2]),\
                        (func[1],func[3],func[4]), \
                        (func[2],func[4],-func[0]-func[3])) ) 
    def S_order(self):
        Order_para=sqrt(inner(self.Q_tensor(self.func_n),self.Q_tensor(self.func_n)))
        self.current_s_order=project(Order_para,self.Scalar_space).compute_vertex_values()
        return self.current_s_order


    def normal_projector(self,alpha,beta,R):
        #r_basis= Expression(("0","x[0]/(a*a)", " x[1]/(b*b)", " x[2]/(c*c)"),a=beta*R,b=beta*R,c=alpha*R, degree=1)
        r_basis_t= Expression(("x[0]/(a*a)", " x[1]/(b*b)", " x[2]/(c*c)","0","0"),a=beta*R,b=beta*R,c=alpha*R, degree=1)

        projected_vector=project(r_basis_t,self.V)
        self.nomral_vector=as_vector([projected_vector[0],projected_vector[1],projected_vector[2]])
        self.Normal_project=Identity(3)-outer(self.nomral_vector,self.nomral_vector)

    def normal_projector_c(self,R):
        #r_basis= Expression(("0","x[0]/(a*a)", " x[1]/(b*b)", " x[2]/(c*c)"),a=beta*R,b=beta*R,c=alpha*R, degree=1)
        r_basis_t= Expression(("x[0]/R", " x[1]/R", " 0","0","0"),R=R, degree=1)

        projected_vector=project(r_basis_t,self.V)
        self.nomral_vector=as_vector([projected_vector[0],projected_vector[1],projected_vector[2]])
        self.Normal_project=Identity(3)-outer(self.nomral_vector,self.nomral_vector)

    def general_normal_projector(self,n_expression):
        projected_vector=project(n_expression,self.V)
        self.nomral_vector=as_vector([projected_vector[0],projected_vector[1],projected_vector[2]])
        self.Normal_project=Identity(3)-outer(self.nomral_vector,self.nomral_vector)


    def Q_tensor_np(self,dir_density=10):
        field_cood=self.func_n.compute_vertex_values()
        field_size=int(len(field_cood)/self.mix_order )
        size_l=len(field_cood[0:field_size][::dir_density])
        full_dir_val=np.ones( (size_l,5)).T
        dir_list = np.zeros((size_l, 3)).T
        #ev_list = np.zeros(size_l)
        print("Dir density------")
        print(size_l)

        for i in range(0,5):
            start_ele_1=int(field_size*i)
            end_ele_1=int(field_size*(i+1))
            full_dir_val[i,:]=field_cood[start_ele_1:end_ele_1][::dir_density]
 
        for j in range(0,size_l):
            v=full_dir_val[:,j]
            Q_np=np.array(((v[0],v[1],v[2]),
                     (v[1],v[3],v[4]),
                     (v[2],v[4],-v[0]-v[3])))
            vals, vecs = np.linalg.eig(Q_np)
            dir_list[:,j][0] = vecs[:,np.argmax(vals)][0]
            dir_list[:,j][1] = vecs[:,np.argmax(vals)][1]
            dir_list[:,j][2] = vecs[:,np.argmax(vals)][2]
         #   print(dir_list[:,i].shape)

            #ev_list[i] = vals[np.argmax(vals)]

        #full_dir_val=full_dir_val/np.sqrt(full_dir_val[0,:]**2+full_dir_val[1,:]**2+full_dir_val[2,:]**2)
        #full_S_roder=field_cood[0:field_size]
        print(dir_list)
        return dir_list#,ev_list#,full_S_roder



    def project_to_function_space(self,Current_Expression):
        self.func_n=project(Current_Expression,self.V) 
        #print('-----')
        #print(np.array(self.func_n.vector())) 


        
    def add_noise(self,nosie_level=0.01,seed=2021):
        func_n_vector_array=np.array(self.func_n.vector())
        np.random.seed(seed)
        nosie=nosie_level*(2*np.random.rand(len(func_n_vector_array))-1)
        func_n_vector_array_w_noise=func_n_vector_array+nosie
        #print(np.array(self.func_n.vector()))
        self.func_n.vector().set_local(func_n_vector_array_w_noise)
        #print(np.array(self.func_n.vector()))
 
    def defect_loc_arry(self,arry_list,data):
        return np.concatenate((arry_list, data), axis=0)

    def defect_check(self,checked_array,checked_array2,target=0.1):
        check_val_max=(np.max(checked_array)-np.max(checked_array2))**2
        check_val_min=(np.min(checked_array)-np.min(checked_array2))**2
        print('-------------------')
        print('Max Difference')
        print(check_val_max)
        print('-------------------')
        print('Min Difference')
        print(check_val_min)
        if (check_val_min+check_val_max)<target:
            return True
        else:
            return False

    def uniform_grad(self,wnt_strength=1,z0=1):
 
        self.effect_range_p=Expression( ('cos(2*atan2(x[1],x[0]+0.01) )'), z0=z0, degree=1)
        self.effect_range_q=Expression( (' sin(2*atan2(x[1],x[0]+0.01) )'), z0=z0, degree=1)
        self.effect_strength=Expression('(x[2]>z0)  ? wnt_strength:0.01', z0=z0,wnt_strength=wnt_strength, degree=1)

    def get_defect_loc(self,mesh_cood,max_val=0.1):
        current_s_order=(self.current_s_order<max_val)
        s_val=self.current_s_order[current_s_order]
        mesh_loc=mesh_cood[current_s_order]
        try:
            kmeans = KMeans(n_clusters=4)
            kmeans.fit(mesh_loc)
            kmean_center=kmeans.cluster_centers_
        except:
            kmean_center=np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])

 
        return s_val,mesh_loc,kmean_center

    def save(self,array,index_arry,append_=True,data_file='peak_data.csv'):
        
        x12=np.sqrt(np.sum((array[0,:]-array[1,:])**2))
        x13=np.sqrt(np.sum((array[0,:]-array[2,:])**2))
        x14=np.sqrt(np.sum((array[0,:]-array[3,:])**2))
        x23=np.sqrt(np.sum((array[1,:]-array[2,:])**2))
        x24=np.sqrt(np.sum((array[1,:]-array[3,:])**2))
        x34=np.sqrt(np.sum((array[2,:]-array[3,:])**2))
        distant_arry=np.array([x12,x13,x14,x23,x24,x34])
        distant_arry=np.sort(distant_arry)
        distant_arry=np.concatenate((index_arry,distant_arry))
        print(distant_arry)
        #ij_distance=np.sqrt((kmean_center[:,0][i]-kmean_center[:,0][j])**2+(kmean_center[:,1][i]-kmean_center[:,1][j])**2+(kmean_center[:,2][i]-kmean_center[:,2][j])**2)

 
        if append_:
            with open(data_file, "a") as f:
                np.savetxt(f, np.column_stack(distant_arry), fmt='%f', delimiter=',')
        else:
            with open(data_file, "w+") as f:
                np.savetxt(f, np.column_stack(distant_arry), fmt='%f', delimiter=',')


    def get_r0_loc(self,mesh_cood,d_fixed,alpha,R):
        darry=np.round((mesh_cood[:,0]**2+mesh_cood[:,1]**2+(mesh_cood[:,2]-alpha*R)**2),1)
        try:
            con_array=np.where(darry==d_fixed)[0]
        except:
            con_array=np.where(np.abs(darry-d_fixed)<0.3)[0]
        d_x=mesh_cood[:,0][con_array[0]]
        d_y=mesh_cood[:,1][con_array[0]]#mesh_cood[:,1][darry==d_fixed]
        d_z=mesh_cood[:,2][con_array[0]]#mesh_cood[:,2][darry==d_fixed]
        target_d=np.sqrt(d_x**2+d_y**2+(d_z)**2)
       # print((d_x**2+d_y**2+(d_z-alpha*R)**2))
       # print(d_fixed)
        sigma=np.sqrt(d_x**2+d_y**2)

        return sigma,d_z


    def average_angle(self,mesh_cood,Dir_profile,band_wdith):
        near_eq=(mesh_cood[:,2]<band_wdith) & (mesh_cood[:,2]>-band_wdith)

        current_dir_x=Dir_profile[0,:][near_eq]
        current_dir_y=Dir_profile[1,:][near_eq]
        current_dir_z=Dir_profile[2,:][near_eq]
        x=mesh_cood[:,0][near_eq]
        y=mesh_cood[:,1][near_eq]
        z=mesh_cood[:,2][near_eq]
        local_phi=np.arctan2(y,x)
        phi_x=-np.sin(local_phi)
        phi_y=np.cos(local_phi)
        #chirality=np.sum((current_dir_x*phi_x+current_dir_y*phi_y)**2)/len(phi_x)
        chirality=np.arccos(np.sum(current_dir_x*phi_x+current_dir_y*phi_y)/len(phi_x))

        #chirality=np.sum((current_dir_x-x)**2+(current_dir_y-y)**2+(current_dir_z-z)**2)/np.sum(x**2+y**2+z**2)
        print('aaa')
        print(chirality)

    def Gradient(self,sigma,R,type="uniform",alpha=1,unit_center_angle=[1,0,0]):
        x, y,z= symbols('x[0], x[1], x[2]')
        # Sympy expression
        e_x=unit_center_angle[0]
        e_y=unit_center_angle[1]
        e_z=unit_center_angle[2]
        print(unit_center_angle)
        R_x=np.array([[1,0,0],[0,np.cos(e_x),np.sin(e_x)],[0,-np.sin(e_x),np.cos(e_x)]])
        R_y=np.array([[np.cos(e_y),0,-np.sin(e_y)],[0,1,0],[np.sin(e_y),0,np.cos(e_y)]])
        R_z=np.array([[np.cos(e_z),np.sin(e_z),0],[-np.sin(e_z),np.cos(e_z),0],[0,0,1]])  
        R_mat=np.matmul(np.matmul(R_x,R_y),R_z)
        A=np.matmul(R_mat,np.array([[x],[y],[z]]))
        B=np.matmul(R_mat,np.array([[R],[R],[alpha*R]]))
        directional_vector=np.matmul(R_mat,np.array([[0],[0],[1]]))
        theta_0=np.arccos((directional_vector[2]/(alpha*R)))[0]
        phi_t=np.arctan2(directional_vector[1],directional_vector[0])[0]

        print(theta_0)
        print(phi_t)

        print('Rotation Matrix')
        print(R_mat)
        x_var=A[0][0]
        y_var=A[1][0]
        z_var=A[2][0]
        print('sizes')
        print(directional_vector)
        x_add_shift=0#np.sin(theta_0)*np.cos(phi_t)#0#B[0][0]
        y_add_shift=0#R*np.sin(theta_0)*np.sin(phi_t)#B[1][0]
        z_add_shift=R*np.sin(theta_0)+alpha*R*np.cos(theta_0)#alpha*R#np.sqrt(np.sum(B**2))#alpha*R#B[2][0]
        #band_cood=(((x_var-x_add_shift)**2+(y_var-y_add_shift)**2+(z_var-z_add_shift)**2)==sigma**2)

        if type=="uniform": # sigma will be act as z0 for uniform 
            f=sp.sqrt((x_var+0.01)**2+(y_var+0.01)**2+(z_var)**2)
            step_fuc=Expression('(x[2]>z0)  ? 1:0.01', z0=sigma, degree=1)
            total_mag=sp.sqrt(dx_f**2+dy_f**2+dz_f**2)
        if  type=="gauss":
            f = sp.exp(-((x_var-x_add_shift)**2+(y_var-y_add_shift)**2+(z_var-z_add_shift)**2)/(2*sigma**2))

            step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
            total_mag=1
        else:
            pass
        # Derivative
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        dz_f = f.diff(z, 1)
        
        
        dx_fE= step_fuc*Expression(ccode(dx_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dy_fE= step_fuc*Expression(ccode(dy_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dz_fE= step_fuc*Expression(ccode(dz_f/total_mag).replace('M_PI', 'pi'),degree=1)
        self.Grad_vector=as_vector((dx_fE,dy_fE,dz_fE))
        f_expresion= step_fuc*Expression(ccode(f).replace('M_PI', 'pi'),degree=1)
        
        dx_f=project(dx_fE,self.Scalar_space).compute_vertex_values()
        dy_f=project(dy_fE,self.Scalar_space).compute_vertex_values()
        dz_f=project(dz_fE,self.Scalar_space).compute_vertex_values()
 
        full_dir_val=np.zeros(self.sys_size).T
 
        full_dir_val[0,:]=dx_f
        full_dir_val[1,:]=dy_f
        full_dir_val[2,:]=dz_f



        return project(f_expresion,self.Scalar_space).compute_vertex_values(),full_dir_val, R_mat


    def Gradient_new(self,sigma,R,alpha=1):
        x, y,z= symbols('x[0], x[1], x[2]')
        # Sympy expression
        z_add_shift=R*alpha
        f = sp.exp(-((x)**2+(y)**2+(z-z_add_shift)**2)/(2*sigma**2))

        step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
        total_mag=1

        # Derivative
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        dz_f = f.diff(z, 1)
        
        mag_=sp.sqrt(dx_f**2+dy_f**2+dz_f**2)
        self.grad_expression=Expression(ccode(mag_).replace('M_PI', 'pi'),degree=1)
        dx_fE= step_fuc*Expression(ccode(dx_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dy_fE= step_fuc*Expression(ccode(dy_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dz_fE= step_fuc*Expression(ccode(dz_f/total_mag).replace('M_PI', 'pi'),degree=1)
        self.Grad_vector=as_vector((dx_fE,dy_fE,dz_fE))
        f_expresion= step_fuc*Expression(ccode(f).replace('M_PI', 'pi'),degree=1)
        
        dx_f=project(dx_fE,self.Scalar_space).compute_vertex_values()
        dy_f=project(dy_fE,self.Scalar_space).compute_vertex_values()
        dz_f=project(dz_fE,self.Scalar_space).compute_vertex_values()
 
        full_dir_val=np.zeros(self.sys_size).T
 
        full_dir_val[0,:]=dx_f
        full_dir_val[1,:]=dy_f
        full_dir_val[2,:]=dz_f



        return project(f_expresion,self.Scalar_space).compute_vertex_values(),full_dir_val
    def get_grad_mag(self):
        return project(self.grad_expression,self.Scalar_space).compute_vertex_values()
    def Gradient_new_two(self,sigma,R,alpha=1,side_sigma=1):
        x, y,z= symbols('x[0], x[1], x[2]')
        # Sympy expression
        z_add_shift=R*alpha
        x_add_shift=R
        f1 = sp.exp(-((x)**2+(y)**2+(z-z_add_shift)**2)/(2*sigma**2))
        f =f1+sp.exp(-((x-x_add_shift)**2+(y)**2+(z)**2)/(2*side_sigma**2))

        step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
        total_mag=1

        # Derivative
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        dz_f = f.diff(z, 1)
        
        mag_=sp.sqrt(dx_f**2+dy_f**2+dz_f**2)
        self.grad_expression=Expression(ccode(mag_).replace('M_PI', 'pi'),degree=1)
        dx_fE= step_fuc*Expression(ccode(dx_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dy_fE= step_fuc*Expression(ccode(dy_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dz_fE= step_fuc*Expression(ccode(dz_f/total_mag).replace('M_PI', 'pi'),degree=1)
        self.Grad_vector=as_vector((dx_fE,dy_fE,dz_fE))
        f_expresion= step_fuc*Expression(ccode(f).replace('M_PI', 'pi'),degree=1)
        
        dx_f=project(dx_fE,self.Scalar_space).compute_vertex_values()
        dy_f=project(dy_fE,self.Scalar_space).compute_vertex_values()
        dz_f=project(dz_fE,self.Scalar_space).compute_vertex_values()
 
        full_dir_val=np.zeros(self.sys_size).T
 
        full_dir_val[0,:]=dx_f
        full_dir_val[1,:]=dy_f
        full_dir_val[2,:]=dz_f



        return project(f_expresion,self.Scalar_space).compute_vertex_values(),full_dir_val


    def Gradient_new_two_cyl(self,sigma,R,alpha=1,side_sigma=1,side_loc=1,z0_shift=10):
        x, y,z= symbols('x[0], x[1], x[2]')
        # Sympy expression
        z_add_shift=R*alpha
        x_add_shift=side_loc
        f1 = sp.exp(-((x)**2+(y)**2+(z-z_add_shift)**2)/(2*sigma**2))
        f =f1+sp.exp(-((x-x_add_shift)**2+(y)**2+(z-z0_shift)**2)/(2*side_sigma**2))

        step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
        total_mag=1

        # Derivative
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        dz_f = f.diff(z, 1)
        
        mag_=sp.sqrt(dx_f**2+dy_f**2+dz_f**2)
        self.grad_expression=Expression(ccode(mag_).replace('M_PI', 'pi'),degree=1)
        dx_fE= step_fuc*Expression(ccode(dx_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dy_fE= step_fuc*Expression(ccode(dy_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dz_fE= step_fuc*Expression(ccode(dz_f/total_mag).replace('M_PI', 'pi'),degree=1)
        self.Grad_vector=as_vector((dx_fE,dy_fE,dz_fE))
        f_expresion= step_fuc*Expression(ccode(f).replace('M_PI', 'pi'),degree=1)
        
        dx_f=project(dx_fE,self.Scalar_space).compute_vertex_values()
        dy_f=project(dy_fE,self.Scalar_space).compute_vertex_values()
        dz_f=project(dz_fE,self.Scalar_space).compute_vertex_values()
 
        full_dir_val=np.zeros(self.sys_size).T
 
        full_dir_val[0,:]=dx_f
        full_dir_val[1,:]=dy_f
        full_dir_val[2,:]=dz_f



        return project(f_expresion,self.Scalar_space).compute_vertex_values(),full_dir_val
         
    def Gradient_two(self,sigma,R,type="gauss",alpha=1,unit_center_angle=[1,0,0],sigma2=0,unit_center_angle2=[1,0,0]):
        x, y,z= symbols('x[0], x[1], x[2]')
        # Sympy expression
        e_x=unit_center_angle[0]
        e_y=unit_center_angle[1]
        e_z=unit_center_angle[2]
        print(unit_center_angle)
        R_x=np.array([[1,0,0],[0,np.cos(e_x),np.sin(e_x)],[0,-np.sin(e_x),np.cos(e_x)]])
        R_y=np.array([[np.cos(e_y),0,-np.sin(e_y)],[0,1,0],[np.sin(e_y),0,np.cos(e_y)]])
        R_z=np.array([[np.cos(e_z),np.sin(e_z),0],[-np.sin(e_z),np.cos(e_z),0],[0,0,1]])  
        R_mat=np.matmul(np.matmul(R_x,R_y),R_z)
        A=np.matmul(R_mat,np.array([[x],[y],[z]]))
        B=np.matmul(R_mat,np.array([[R],[R],[alpha*R]]))
        directional_vector=np.matmul(R_mat,np.array([[0],[0],[1]]))
        theta_0=np.arccos((directional_vector[2]/(alpha*R)))[0]
        phi_t=np.arctan2(directional_vector[1],directional_vector[0])[0]

        print(theta_0)
        print(phi_t)

        print('Rotation Matrix')
        print(R_mat)
        x_var=A[0][0]
        y_var=A[1][0]
        z_var=A[2][0]
        print('sizes')
        print(directional_vector)
        x_add_shift=0#np.sin(theta_0)*np.cos(phi_t)#0#B[0][0]
        y_add_shift=0#R*np.sin(theta_0)*np.sin(phi_t)#B[1][0]
        z_add_shift=R*np.sin(theta_0)+alpha*R*np.cos(theta_0)#alpha*R#np.sqrt(np.sum(B**2))#alpha*R#B[2][0]
        #band_cood=(((x_var-x_add_shift)**2+(y_var-y_add_shift)**2+(z_var-z_add_shift)**2)==sigma**2)


        f1 = sp.exp(-((x_var-x_add_shift)**2+(y_var-y_add_shift)**2+(z_var-z_add_shift)**2)/(2*sigma**2))

        step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
        total_mag=1


        # Sympy expression
        e_x=unit_center_angle2[0]
        e_y=unit_center_angle2[1]
        e_z=unit_center_angle2[2]
        print(unit_center_angle2)
        R_x=np.array([[1,0,0],[0,np.cos(e_x),np.sin(e_x)],[0,-np.sin(e_x),np.cos(e_x)]])
        R_y=np.array([[np.cos(e_y),0,-np.sin(e_y)],[0,1,0],[np.sin(e_y),0,np.cos(e_y)]])
        R_z=np.array([[np.cos(e_z),np.sin(e_z),0],[-np.sin(e_z),np.cos(e_z),0],[0,0,1]])  
        R_mat=np.matmul(np.matmul(R_x,R_y),R_z)
        A=np.matmul(R_mat,np.array([[x],[y],[z]]))
        B=np.matmul(R_mat,np.array([[R],[R],[R]]))
        directional_vector=np.matmul(R_mat,np.array([[0],[0],[1]]))
        theta_0=np.arccos((directional_vector[2]/(alpha*R)))[0]
        phi_t=np.arctan2(directional_vector[1],directional_vector[0])[0]

        print(theta_0)
        print(phi_t)

        print('Rotation Matrix')
        print(R_mat)
        x_var=A[0][0]
        y_var=A[1][0]
        z_var=A[2][0]
        print('sizes')
        print(directional_vector)
        x_add_shift=0#np.sin(theta_0)*np.cos(phi_t)#0#B[0][0]
        y_add_shift=0#R*np.sin(theta_0)*np.sin(phi_t)#B[1][0]
        z_add_shift=R*np.sin(theta_0)+alpha*R*np.cos(theta_0)#alpha*R#np.sqrt(np.sum(B**2))#alpha*R#B[2][0]
        #band_cood=(((x_var-x_add_shift)**2+(y_var-y_add_shift)**2+(z_var-z_add_shift)**2)==sigma**2)



        f =f1+ sp.exp(-((x_var-x_add_shift)**2+(y_var-y_add_shift)**2+(z_var-z_add_shift)**2)/(2*sigma2**2))

        step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
        total_mag=1

        # Derivative
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        dz_f = f.diff(z, 1)
        
        
        dx_fE= step_fuc*Expression(ccode(dx_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dy_fE= step_fuc*Expression(ccode(dy_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dz_fE= step_fuc*Expression(ccode(dz_f/total_mag).replace('M_PI', 'pi'),degree=1)






        self.Grad_vector=as_vector((dx_fE,dy_fE,dz_fE))
        f_expresion= step_fuc*Expression(ccode(f).replace('M_PI', 'pi'),degree=1)
        
        dx_f=project(dx_fE,self.Scalar_space).compute_vertex_values()
        dy_f=project(dy_fE,self.Scalar_space).compute_vertex_values()
        dz_f=project(dz_fE,self.Scalar_space).compute_vertex_values()
 
        full_dir_val=np.zeros(self.sys_size).T
 
        full_dir_val[0,:]=dx_f
        full_dir_val[1,:]=dy_f
        full_dir_val[2,:]=dz_f



        return project(f_expresion,self.Scalar_space).compute_vertex_values(),full_dir_val, R_mat


    



    def alignment_tensor(self):
         return outer(self.Grad_vector,self.Grad_vector)- tr(outer(self.Grad_vector,self.Grad_vector))*Identity(3)
    
    def director(self):
        Q_tensor_np=project(self.Q_tensor(self.func_n) ,self.T).compute_vertex_values()
        #Q_tensor_np=project(self.func_n ,self.V).compute_vertex_values()
        print(Q_tensor_np)
        print(Q_tensor_np.shape)
    def normal_error(self):
        Q=self.Q_tensor(self.func_n)
        n=self.nomral_vector
        normalForce = outer(dot(Q, n), n) + outer(n, dot(Q, n)) - 0.5 * inner(outer(n,n), Q) * outer(n, n)
        return project(sqrt(inner(normalForce,normalForce)),self.Scalar_space).compute_vertex_values()

    def variational_nematic(self,dt,K=1,const=2,wnt_strength=1,Wnt_on=True):
        dx=self.dx
        Q=self.Q_tensor(self.func)
        Q_n=self.Q_tensor(self.func_n)
        Q_test=self.Q_tensor(self.test_func)
        n=self.nomral_vector
        
        normalForce = outer(dot(Q, n), n) + outer(n, dot(Q, n)) - 0.5 * inner(outer(n,n), Q) * outer(n, n)

        self.k=Constant(dt)
        penalty_strength = Constant(10000) # Normal penalty

        """    
        
        gradx=self.gradx
        grady=self.grady
        gradz=self.gradz"""
        C_const=Constant(const)
        F_time =inner((Q  - Q_n),Q_test)/ self.k*dx
        p= Index()
        q= Index()
        r= Index()
        s= Index()
        k= Index()

           # K * inner(nabla_grad(Q), nabla_grad(Q_test))*dx + \
        P_j=Identity(3)-outer(n,n)
        F_nematic = -inner((1 - C_const*inner(Q,Q))*Q, Q_test)*dx +\
            K*P_j[p,q]*P_j[r,s]*grad(Q)[q,s,k]*grad(Q_test)[p,r,k]*dx+\
            penalty_strength * inner(normalForce, Q_test)*dx 
   #     F_nematic = -inner((1 - C_const*inner(Q,Q))*Q, Q_test)*dx +\
    #        K*normalForce[p,q]*normalForce[r,s]*nabla_grad(Q)[p,s,k]*nabla_grad(Q_test)[r,q,k]*dx+\
     #       penalty_strength * inner(normalForce, Q_test)*dx 

    #    F_nematic = -inner((1 - C_const*inner(Q,Q))*Q, Q_test)*dx +\
      #      K*normalForce[p,s]*normalForce[r,q]*nabla_grad(Q)[p,s,k]*nabla_grad(Q_test)[r,q,k]*dx+\
      #      penalty_strength * inner(normalForce, Q_test)*dx 

      #  F_nematic = -inner((1 - C_const*inner(Q,Q))*Q, Q_test)*dx +\
       #     K*nabla_grad(Q)[q,s,k]*nabla_grad(Q_test)[q,s,k]*dx+\
        #    penalty_strength * inner(normalForce, Q_test)*dx 

        Q_align=self.alignment_tensor()

        if Wnt_on:
            wnt_strength=wnt_strength
        else: 
            wnt_strength=0
        #print(wnt_strength)
        Falign=Constant(wnt_strength) * inner(Q_align, Q_test)*dx 
        self.F=F_time+F_nematic+Falign
        dQVec = TrialFunction(self.V)
        self.J = derivative(self.F, self.func, dQVec)

    def get_element(self,index_loc,ele_type='func'):
        if ele_type=='func':
            return self.func[str(index_loc)]
        elif ele_type=='test_fuc':
            return self.test_func[str(index_loc)] 
        else:
            return self.func_n[str(index_loc)] 

    def evolve(self,dt,J=0,noise_level=0,apply_bc=True):
        self.k.assign(dt)
        F=self.F
        u=self.func
        nonlin_problem = NonlinearVariationalProblem(F, u, J=self.J)
        nonlin_solver  = NonlinearVariationalSolver(nonlin_problem)
        #self.config_newton_solver(nonlin_solver, True)
        prm = nonlin_solver.parameters

        prm['nonlinear_solver'] = 'newton'
        prm['newton_solver']['linear_solver'] = 'gmres'
        prm['newton_solver']['preconditioner'] = 'ilu'
        prm['newton_solver']['absolute_tolerance'] = 1E-9
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 100000
        nonlin_solver.solve()
        #solve(F==0,u,J=self.J,solver_parameters={"newton_solver":{"relative_tolerance":1e-10},"newton_solver":{"maximum_iterations":300}})
        self.u=u
        
        self.func_n.assign(u)

    # Solver options
    def config_newton_solver(self,solver, iterative_solver=True):
        
        prm = solver.parameters
      #  prm['newton_solver']['absolute_tolerance'] = 1E-8
       # prm['newton_solver']['relative_tolerance'] = 1E-7
       # prm['newton_solver']['maximum_iterations'] = 25
       # prm['newton_solver']['relaxation_parameter'] = 1.0
        if iterative_solver:
            prm['nonlinear_solver'] = 'newton'
            prm['newton_solver']['linear_solver'] = 'mumps'
            #prm['newton_solver']['preconditioner'] = 'ilu'
            prm['newton_solver']['absolute_tolerance'] = 1E-9
            prm['newton_solver']['relative_tolerance'] = 1E-7
            prm['newton_solver']['maximum_iterations'] = 1000
 
