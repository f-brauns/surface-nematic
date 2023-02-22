# %% Importing
#########################################################
#########################################################

from audioop import avgpp
from math import atan2
from sample_script import *
from ufl import atan_2
import pandas as pd 
import csv 
class SymTransZ(Expression):
    'Given u: (x, y) --> R create v: (x, y, z) --> R, v(x, y, z) = u(x, y).'
    def __init__(self, u):
        self.u = u





class Fenics_function_space:
    ''' Process the gmesh file into data array, and export into a .dat format'''
    def __init__(self,mesh,sys_size,finite_element,mix_order=1,file_dir='/home/zhwang/hydra_fenics/nematic_project/cache/nematic_fig_2cp2'):
        self.mesh=mesh
        self.sys_size=sys_size
        self.x, self.y = SpatialCoordinate(mesh)
        self.mix_order=mix_order

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

        # Define trial and test functions. 
        self.func = Function(self.V)
        self.test_func=TestFunction(self.V)

        q_degree=2
        dx=Measure("dx",domain=mesh)
        dx=dx(metadata={'quadrature_degree': q_degree})
        self.dx=dx
 
        print(os.getcwd())
        # stepping 
        t = float(0)
        iii=0 # stepper 
        files=[] # file list 

    def Polar_vector(self,func):
        return as_vector((func[0],func[1]))
    def S_order_polar(self):
        Order_para=sqrt(inner(self.Polar_vector(self.func_n),self.Polar_vector(self.func_n)))
        self.current_s_order_polar=project(Order_para,self.Scalar_space).compute_vertex_values()
        return self.current_s_order_polar

    def Polar_vector_np(self,dir_density=10):
        field_cood=self.func_n.compute_vertex_values()
        field_size=int(len(field_cood)/self.mix_order )
        size_l=len(field_cood[0:field_size][::dir_density])
        full_dir_val=np.ones( (size_l,self.mix_order)).T
        dir_list = np.zeros((size_l, 2)).T
        #ev_list = np.zeros(size_l)
        print("Dir density------")
        print(size_l)

        for i in range(0,self.mix_order):
            start_ele_1=int(field_size*i)
            end_ele_1=int(field_size*(i+1))
            full_dir_val[i,:]=field_cood[start_ele_1:end_ele_1][::dir_density]

 
        for j in range(0,size_l):
            v=full_dir_val[:,j]
            Q_np=np.array(((v[0],v[1]),
                     (v[1],-v[0])))
            vals, vecs = np.linalg.eig(Q_np)
            dir_list[:,j][0] = vecs[:,np.argmax(vals)][0]
            dir_list[:,j][1] = vecs[:,np.argmax(vals)][1]
            dir_list[:,j][0]=dir_list[:,j][0] /np.sqrt(vecs[:,np.argmax(vals)][0]**2+vecs[:,np.argmax(vals)][1]**2)
            dir_list[:,j][1]=dir_list[:,j][1] /np.sqrt(vecs[:,np.argmax(vals)][0]**2+vecs[:,np.argmax(vals)][1]**2)
        return dir_list#,ev_list#,full_S_roder

    def Q_tensor(self,func):
         return as_matrix(((func[0],func[1]),\
                        (func[1],-func[0])) ) 

    def S_order(self):
        Order_para=sqrt(inner(self.Q_tensor(self.func_n),self.Q_tensor(self.func_n)))
        self.current_s_order=project(Order_para,self.Scalar_space).compute_vertex_values()
        return self.current_s_order

    def Q_tensor_np(self,dir_density=10):
        field_cood=self.func_n.compute_vertex_values()
        field_size=int(len(field_cood)/self.mix_order )
        size_l=len(field_cood[0:field_size][::dir_density])
        full_dir_val=np.ones( (size_l,self.mix_order)).T
        dir_list = np.zeros((size_l, 2)).T
        #ev_list = np.zeros(size_l)
        print("Dir density------")
        print(size_l)

        for i in range(0,self.mix_order):
            start_ele_1=int(field_size*i)
            end_ele_1=int(field_size*(i+1))
            full_dir_val[i,:]=field_cood[start_ele_1:end_ele_1][::dir_density]

 
        for j in range(0,size_l):
            v=full_dir_val[:,j]
            Q_np=np.array(((v[0],v[1]),
                     (v[1],-v[0])))
            vals, vecs = np.linalg.eig(Q_np)
            dir_list[:,j][0] = vecs[:,np.argmax(vals)][0]
            dir_list[:,j][1] = vecs[:,np.argmax(vals)][1]
            dir_list[:,j][0]=dir_list[:,j][0] /np.sqrt(vecs[:,np.argmax(vals)][0]**2+vecs[:,np.argmax(vals)][1]**2)
            dir_list[:,j][1]=dir_list[:,j][1] /np.sqrt(vecs[:,np.argmax(vals)][0]**2+vecs[:,np.argmax(vals)][1]**2)
        return dir_list#,ev_list#,full_S_roder



    def project_to_function_space(self,Current_Expression):
        self.func_n=project(Current_Expression,self.V) 
        #print('-----')
        #print(np.array(self.func_n.vector())) 


        
    def add_noise(self,nosie_level=0.01,seed=2022):
        func_n_vector_array=np.array(self.func_n.vector())
        np.random.seed(seed)
        nosie=nosie_level*(2*np.random.rand(len(func_n_vector_array))-1)
        func_n_vector_array_w_noise=func_n_vector_array+nosie
        #print(np.array(self.func_n.vector()))
        self.func_n.vector().set_local(func_n_vector_array_w_noise)
        #print(np.array(self.func_n.vector()))
 
    def defect_loc_arry(self,arry_list,data):
        print(np.concatenate((arry_list, data), axis=0))
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

    def defect_out(self,row_ele,col_ele,row_ele_index=0,col_ele_index=0,checked_array=[],csv_path='../path.csv'):


        with open(csv_path) as fl:
            r = csv.reader(fl) # Here your csv file
            lines = list(r)
            sort_arry_ele=np.sum(-np.sort(-checked_array[::2])[0:4])
            lines[row_ele_index][col_ele_index]=sort_arry_ele
            print(sort_arry_ele)
            writer = csv.writer(open(csv_path, 'w'))
            writer.writerows(lines)
        #pd.DataFrame(checked_array[::2]).to_csv(csv_path,mode='a', header=None, index=None)
    def defect_bare_out(self,row_ele,col_ele,row_ele_index=0,col_ele_index=0,checked_array=[],csv_path='../path.csv'):

        with open(csv_path) as fl:
            r = csv.reader(fl) # Here your csv file
            lines = list(r)
            print(np.sort(-checked_array))
            print('DDDD')
            sort_arry_ele=-np.sort(-checked_array)[0]
            lines[row_ele_index][col_ele_index]=sort_arry_ele
            print(sort_arry_ele)
            writer = csv.writer(open(csv_path, 'w'))
            writer.writerows(lines)

    def get_defect_loc(self,mesh_cood,max_val=0.4):
        current_s_order=(self.current_s_order<max_val)
        s_val=self.current_s_order[current_s_order]
        mesh_loc=mesh_cood[current_s_order]
        try:
            kmeans = KMeans(n_clusters=4)
            kmeans.fit(mesh_loc)
            kmean_center=kmeans.cluster_centers_
            print(kmean_center)
        except:
            kmean_center=np.array([[0,0],[0,0],[0,0],[0,0]])

 
        return s_val,mesh_loc,kmean_center





    def Gradient(self,sigma,type="gauss",aspect_ratio=1):
        x, y= symbols('x[0], x[1]')
        # Sympy expression
        if  type=="gauss":
            f = sp.exp(-((x)**2+(aspect_ratio*y)**2)/(2*(sigma*sigma)))

            step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
            total_mag=1
        elif type=="exp":
            f = sp.exp(-sp.sqrt((x)**2+(aspect_ratio*y)**2+0.001)/(sigma))

            step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
            total_mag=1

        else:
            pass
        # Derivative
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        
        print(ccode(dx_f/total_mag))
        dx_fE= step_fuc*Expression(ccode(dx_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dy_fE= step_fuc*Expression(ccode(dy_f/total_mag).replace('M_PI', 'pi'),degree=1)

        self.Grad_vector=as_vector((dx_fE,dy_fE))
        f_expresion= step_fuc*Expression(ccode(f).replace('M_PI', 'pi'),degree=1)
        grad_mag=sqrt(dx_fE*dx_fE+dy_fE*dy_fE)

        dx_f=project(dx_fE/grad_mag,self.Scalar_space).compute_vertex_values()
        dy_f=project(dy_fE/grad_mag,self.Scalar_space).compute_vertex_values()
 
        full_dir_val=np.zeros(self.sys_size).T
 
        full_dir_val[0,:]=dx_f
        full_dir_val[1,:]=dy_f

        

        return project(f_expresion,self.Scalar_space).compute_vertex_values(),project(grad_mag,self.Scalar_space).compute_vertex_values(),full_dir_val
    """    
    def Wall_Gradient(self,wall_loc,sigma,type="uniform",aspect_ratio=1):
        x, y= symbols('x[0], x[1]')
        # Sympy expression
        if  type=="gauss":
            f = sp.exp(-((sp.sqrt((x+0.01)**2+(aspect_ratio*y+0.01)**2)-wall_loc)**2)/(2*sigma**2))

            step_fuc=1#Expression('(x[2]>z0)  ? 1:1', z0=sigma, degree=1)
            total_mag=1
        else:
            pass
        # Derivative
        dx_f = f.diff(x, 1)
        dy_f = f.diff(y, 1)
        print(ccode(dx_f/total_mag))
        print(ccode(f))
        dx_fE= step_fuc*Expression(ccode(dx_f/total_mag).replace('M_PI', 'pi'),degree=1)
        dy_fE= step_fuc*Expression(ccode(dy_f/total_mag).replace('M_PI', 'pi'),degree=1)

        self.Grad_vector=as_vector((dx_fE,dy_fE))
        f_expresion= step_fuc*Expression(ccode(f).replace('M_PI', 'pi'),degree=1)
        
        dx_f=project(dx_fE,self.Scalar_space).compute_vertex_values()
        dy_f=project(dy_fE,self.Scalar_space).compute_vertex_values()
 
        full_dir_val=np.zeros(self.sys_size).T
 
        full_dir_val[0,:]=dx_f
        full_dir_val[1,:]=dy_f

        return project(f_expresion,self.Scalar_space).compute_vertex_values(),full_dir_val
        """
    def alignment_tensor(self):
         return outer(self.Grad_vector,self.Grad_vector)- tr(outer(self.Grad_vector,self.Grad_vector))*Identity(2)

    def director(self):
        Q_tensor_np=project(self.Q_tensor(self.func_n) ,self.T).compute_vertex_values()
        #Q_tensor_np=project(self.func_n ,self.V).compute_vertex_values()

    def update_var_arry(self,beta_min,beta_max,beta_resolution,g_min,g_max,g_resolution):
        if beta_min==beta_max:
            beta_arry=beta_min*np.ones(beta_resolution)
        elif g_min==g_max:
            g_arry=g_min*np.ones(g_resolution)
        else:
            beta_arry=np.linspace(beta_min,beta_max,beta_resolution)
            g_arry=np.linspace(g_min,g_max,g_resolution)

        return beta_arry , g_arry
        



        

    def variational_nematic(self,dt,K=1,const=2,wnt_strength=1,Wnt_on=True):
        dx=self.dx
        Q=self.Q_tensor(self.func)
        Q_n=self.Q_tensor(self.func_n)
        Q_test=self.Q_tensor(self.test_func)

        self.k=Constant(dt)

        C_const=Constant(const)
        F_time =inner((Q  - Q_n),Q_test)/ self.k*dx
 

        F_nematic = -inner((1 - C_const*inner(Q,Q))*Q, Q_test)*dx +\
            K * inner(nabla_grad(Q), nabla_grad(Q_test))*dx 

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
        #solve(F==0,u)
         
        nonlin_problem = NonlinearVariationalProblem(F, u, J=self.J)
        nonlin_solver  = NonlinearVariationalSolver(nonlin_problem)
        #self.config_newton_solver(nonlin_solver, True)
        prm = nonlin_solver.parameters

        prm['nonlinear_solver'] = 'newton'
        prm['newton_solver']['linear_solver'] = 'gmres'
        prm['newton_solver']['preconditioner'] = 'icc'
        prm['newton_solver']['absolute_tolerance'] = 1E-9
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 1000
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
 
