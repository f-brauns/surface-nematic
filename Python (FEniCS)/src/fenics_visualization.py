# %% Importing
#########################################################
#########################################################

from sample_script import *

 


class Visualization:
    ''' Process the gmesh file into data array, and export into a .dat format'''
    def __init__(self,full_mesh_cood,def_loc_mesh=1,Dir_mesh=1,R=1,row_num=2,column_num=2,w=30,h=30):
        self.fig = plt.figure()
        self.fig.suptitle('')
        self.fig.set_figheight(h)
        self.fig.set_figwidth(w)
        self.row_num=row_num
        self.column_num=column_num
        total_num=row_num*column_num
        self.axes={"total_num":total_num}
        max_z=np.amax(full_mesh_cood[:,2], axis=0)
        max_x=np.amax(full_mesh_cood[:,0], axis=0)
        self.max_array=np.array([max_z,max_x])
        #max_array=np.amax(max_array, axis=0)
 
        min_z=np.amin(full_mesh_cood[:,2], axis=0)
        min_x=np.amin(full_mesh_cood[:,0], axis=0)
        min_array=np.array([min_z,min_x])
        self.min_array=np.amin(min_array, axis=0)
        self.total_min= np.amin(self.min_array, axis=0) 
        self.total_max= np.amax(self.max_array, axis=0) 
      #  print(max_z)
      #('dahda')


        for i in range(1,total_num+1):
            current_plot_num=int(str(self.row_num)+str(self.column_num)+str(i))
           # print(i)
            self.axes[str(i)] = self.fig.add_subplot(self.row_num, self.column_num,i, projection='3d')
            self.axes[str(i)].set_xlabel('x (xi)')
            self.axes[str(i)].set_ylabel('y (xi)') 
            self.axes[str(i)].set_zlabel('z (xi)')
            #self.axes[str(i)].set_xlim(self.min_array[0],self.max_array[0] )
            #self.axes[str(i)].set_ylim(self.min_array[1],self.max_array[1] )
            #self.axes[str(i)].set_zlim(self.min_array[2],self.max_array[2] )

           # self.axes[str(i)].set_xlim(min_x,max_x)
           # self.axes[str(i)].set_ylim(min_x,max_x )
            self.axes[str(i)].set_xlim(min_z,max_z)
            self.axes[str(i)].set_ylim(min_z,max_z )
            self.axes[str(i)].set_zlim(min_z,max_z )
            self.axes[str(i)].set_box_aspect((1,1,1))



        self.mesh_cood=np.array([full_mesh_cood,def_loc_mesh,Dir_mesh],dtype='object') # 0 is the inner region and 1 is the side region

 
    def view_angle(self,Ct_id=0,view_elev=0, view_azim=0):

        #view_elev=90 # add 90 on theta 
        #view_azim=0 # keep phi the same. effectively, one wants the view axis to be perpendicular to the first cut axis. 
        view_elev=np.deg2rad(90-view_elev)
        view_azim=np.deg2rad(view_azim)
 
        mesh_cood=self.mesh_cood[Ct_id]
         # for point of view rotation. 
        divide_plane_view=((np.sin(view_elev)*np.cos(view_azim)*mesh_cood[:,0]+np.sin(view_elev)*np.sin(view_azim)*mesh_cood[:,1]+np.cos(view_elev)*mesh_cood[:,2]) )>0#  & ((np.sin(elev)*np.cos(azim)*mesh_cood[:,0]+np.sin(elev)*np.sin(azim)*mesh_cood[:,1]+np.cos(elev)*mesh_cood[:,2])>0)
        #dir_mesh=mesh_cood[:,0]>0
        #self.divide_plane_view=divide_plane_view
        return divide_plane_view

    def director_join(self,middle_bulk_bool,side_bulk_bool,bc_bool,inner_dir_val,side_dir_val,bc_dir_val):
        full_dir_val=np.zeros(self.mesh_cood[2].shape).T

        full_dir_val[0,:][middle_bulk_bool]=inner_dir_val[0,:] 
        full_dir_val[1,:][middle_bulk_bool]=inner_dir_val[1,:] 
        full_dir_val[2,:][middle_bulk_bool]=inner_dir_val[2,:] 
 
        full_dir_val[0,:][side_bulk_bool]=side_dir_val[0,:] 
        full_dir_val[1,:][side_bulk_bool]=side_dir_val[1,:] 
        full_dir_val[2,:][side_bulk_bool]=side_dir_val[2,:] 
    
        full_dir_val[0,:][bc_bool]=bc_dir_val[0,:] 
        full_dir_val[1,:][bc_bool]=bc_dir_val[1,:] 
        full_dir_val[2,:][bc_bool]=bc_dir_val[2,:] 

        return full_dir_val
    def plot_marked_region(self,boundary_parts, bulk_parts,plot_index_1=1,view_elev=0, view_azim=0):
        """     
        ax_1= self.axes[str(plot_index_1)]
        ax_1.view_init(view_elev , view_azim)
        ax_1.set_title('Mesh boundary', fontsize=20)
        plot(boundary_parts)"""
        ax_1= self.axes[str(plot_index_1+1)]
        ax_1.view_init(view_elev , view_azim)
        ax_1.set_title('Mesh bulk', fontsize=20)
        plot(bulk_parts)
        plt.show()
        plt.savefig('/test_'+str(0)+'.png', dpi=100)

    def plot_mesh(self,Ct_id=0,plot_index=1,view_elev=0, view_azim=0):
        divide_plane_view=self.view_angle(Ct_id,view_elev, view_azim)
        ax_1= self.axes[str(plot_index)]
        ax_1.view_init(view_elev , view_azim)
        ax_1.set_title('Mesh', fontsize=20)


        mesh_cood=self.mesh_cood[Ct_id]
         
        ax_1.scatter3D(mesh_cood[:,0][divide_plane_view],mesh_cood[:,1][divide_plane_view],mesh_cood[:,2][divide_plane_view])
        #ax_1.scatter3D(mesh_cood[:,0],mesh_cood[:,1],mesh_cood[:,2])

    def plot_mesh_band(self,mesh_cood,plot_index=1,view_elev=0, view_azim=0):
        #view_elev=90 # add 90 on theta 
        #view_azim=0 # keep phi the same. effectively, one wants the view axis to be perpendicular to the first cut axis. 
        view_elev=np.deg2rad(90-view_elev)
        view_azim=np.deg2rad(view_azim)
         # for point of view rotation. 
        divide_plane_view=((np.sin(view_elev)*np.cos(view_azim)*mesh_cood[:,0]+np.sin(view_elev)*np.sin(view_azim)*mesh_cood[:,1]+np.cos(view_elev)*mesh_cood[:,2]) >0)#  & ((np.sin(elev)*np.cos(azim)*mesh_cood[:,0]+np.sin(elev)*np.sin(azim)*mesh_cood[:,1]+np.cos(elev)*mesh_cood[:,2])>0)
        #dir_mesh=mesh_cood[:,0]>0
        ax_1= self.axes[str(plot_index)]
        ax_1.view_init(view_elev , view_azim)
        ax_1.set_title('Mesh Band', fontsize=20)
         
        ax_1.scatter3D(mesh_cood[:,0][divide_plane_view],mesh_cood[:,1][divide_plane_view],mesh_cood[:,2][divide_plane_view])
     
    def add_subplot_axes(self,ax,rect=[0.7,0.7,0.3,0.3],axisbg='white',D3=False):
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position  = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
        if D3==True:
            subax = fig.add_axes([x,y,width,height],facecolor=axisbg, projection='3d')
        else:
            subax = fig.add_axes([x,y,width,height],facecolor=axisbg)

        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.5
        y_labelsize *= rect[3]**0.5
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
        subax.xaxis.set_tick_params(labelsize=x_labelsize)


        return subax

    def plot_func_surface(self,field_cood,Ct_id=0,plot_index=1,title='Top Layer Nematic Field',color_range=[0,1],set_range=True,view_elev=0, view_azim=0,marker_size=27,cbar=True,alpha_c=1,bg_off=False,cmap='afmhot'):
        mesh_cood=self.mesh_cood[Ct_id]
        ax_1= self.axes[str(plot_index)]
#        current_plot_num=int(str(self.row_num)+str(self.column_num)+str(plot_index))
 #       self.ax_1 = self.fig.add_subplot(current_plot_num, projection='3d')
        if set_range==True:
            p3dc=ax_1.scatter3D(mesh_cood[:,0],mesh_cood[:,1] ,mesh_cood[:,2], c=np.round(field_cood,2), s=marker_size,  marker='^',cmap=cmap,alpha=alpha_c,vmin=color_range[0], vmax=color_range[1],zorder=1 ) #, facecolors=cmap(colors), shade=True, alpha=1
            if cbar==True:

                cl=self.fig.colorbar(p3dc, ax=ax_1,label='Order strength ')
                cl.mappable.set_clim(color_range[0],color_range[1])
            else:
                pass
            
        else:
            p3dc=ax_1.scatter3D(mesh_cood[:,0],mesh_cood[:,1] ,mesh_cood[:,2], c=np.round(field_cood,2), s=marker_size,  marker='^',cmap='afmhot',alpha=1,zorder=3) #, facecolors=cmap(colors), shade=True, alpha=1
            if cbar==True:

                cl=self.fig.colorbar(p3dc, ax=ax_1,label='Order strength ')
            else:
                pass

            #cl=self.fig.colorbar(p3dc, ax=ax_1,label='Order strength ')
        #p3dc.set_fc(colors)
        ax_1.view_init(view_elev, view_azim)
        ax_1.set_title(title, fontsize=20)
        if bg_off:
            ax_1.grid(False)
            #plt.axis('off')
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            ax_1.set_zticks([])
            ax_1.set(facecolor = "white")
            ax_1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax_1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax_1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax_1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax_1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax_1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
 

    def plot_defect_center(self,kmean_center,plot_index=1,z0=1,alpha=1,R=1,title='da',color='red',view_elev=0, view_azim=0,sub_fig=False,ring_density=0,sub_on=True):

        ax_1= self.axes[str(plot_index)]
#        current_plot_num=int(str(self.row_num)+str(self.column_num)+str(plot_index))
 #       self.ax_1 = self.fig.add_subplot(current_plot_num, projection='3d')
        theta_0=np.arccos(z0/(alpha*R))
        phi_t=np.linspace(0,2*np.pi,100)
        x=R*np.sin(theta_0)*np.cos(phi_t)
        y=R*np.sin(theta_0)*np.sin(phi_t)
        ax_1.plot3D(x,y,z0,color,zorder=10,lw=5)
        p3dc=ax_1.scatter3D(kmean_center[:,0],kmean_center[:,1] ,kmean_center[:,2], s=400,  marker='o',alpha=1,zorder=10) #, facecolors=cmap(colors), shade=True, alpha=1
        theta_ring=np.linspace(0.2,np.pi-0.2,ring_density)
        for i in range(0,ring_density):
            x=R*np.sin(theta_ring[i])*np.cos(phi_t)
            y=R*np.sin(theta_ring[i])*np.sin(phi_t)
            ax_1.plot3D(x,y,alpha*R*np.cos(theta_ring[i]),'black',zorder=10,alpha=0.3)
        
        #p3dc.set_fc(colors)
        ax_1.view_init(view_elev, view_azim)
        ax_1.set_title(title, fontsize=20)
        if sub_fig:
            a=6#alpha*R
            if sub_on:
                sub_axis=self.add_subplot_axes(ax_1)
                sub_axis.get_xaxis().set_visible(False)
                sub_axis.get_yaxis().set_visible(False)

                sub_axis.text(a,a+0.5,"Defect Distances:", size=20, zorder=10,color='k')
                sub_axis.text(a,-3*a+0.5,"Defect Angle:", size=20, zorder=10,color='k')
                sub_axis.set_xlim(-a,a)
                sub_axis.set_ylim(-a,a)
                sub_axis.axis('off')
                ax_1.set_xlim(-alpha*R,alpha*R)
                ax_1.set_ylim(-alpha*R,alpha*R )
                ax_1.set_zlim(-alpha*R,alpha*R )

            c_num=0
            properties_array=[]
            for i in range(len(kmean_center)):
                txt="Defect: "+str(i)
                ax_1.text(kmean_center[:,0][i]-0.2,kmean_center[:,1][i]-0.2,kmean_center[:,2][i]+0.7,  '%s' % (str(i)), size=20, zorder=1,color='k')

                for j in range(0,len(kmean_center)):
                    if j>i:
                        ij_distance=np.sqrt((kmean_center[:,0][i]-kmean_center[:,0][j])**2+(kmean_center[:,1][i]-kmean_center[:,1][j])**2+(kmean_center[:,2][i]-kmean_center[:,2][j])**2)
                        i_mag=np.sqrt(kmean_center[:,0][i]**2+kmean_center[:,1][i]**2+kmean_center[:,2][i]**2)
                        j_mag=np.sqrt(kmean_center[:,0][j]**2+kmean_center[:,1][j]**2+kmean_center[:,2][j]**2)

                        ij_angle=(180/np.pi)*np.arccos(1/(i_mag*j_mag)*(kmean_center[:,0][i]*kmean_center[:,0][j]+kmean_center[:,1][i]*kmean_center[:,1][j]+kmean_center[:,2][i]*kmean_center[:,2][j]))
                        if sub_on:
                            sub_axis.text(a,a-2*c_num-2, '%s -> %s: %s ' % (str(i),str(j),str(round(ij_distance,2))), size=20, zorder=1,color='k')
                            sub_axis.text(a,-3*a-2*c_num-2, '%s -> %s: %s ' % (str(i),str(j),str(round(ij_angle,2))), size=20, zorder=1,color='k')
                        c_num=c_num+1
                        properties_array.append(ij_distance)
                        properties_array.append(ij_angle)


                    else:
                        pass

            #sub_axis.set_title('Defect Distance: '+str(alpha), fontsize=20)




        else:
            pass
        return np.array([properties_array])
        

    def plot_axis(self,view_elev=0, view_azim=0,plot_index=1):
        ax_1= self.axes[str(plot_index)]
        sub_axis=self.add_subplot_axes(ax_1,D3=True)
        sub_axis.get_xaxis().set_visible(False)
        sub_axis.get_yaxis().set_visible(False)

        sub_axis.set_title('Aspect Ratio: ', fontsize=20)
        sub_axis.set_xlim(0,1)
        sub_axis.set_ylim(0,1)
        sub_axis.set_zlim(0,1)
        zero_pt=[0,0,0]
        x_point=[1,0,0]
        y_point=[0,1,0]
        z_point=[0,0,1]
        sub_axis.quiver(0,0,0,1, 0,0,color = 'red', lw = 3)
        #a_x = plt.Arrow3D([0, 1], [0,0], [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        #sub_axis.add_artist(a_x)
        sub_axis.view_init(view_elev, view_azim)


    def plot_wnt_range(self,plot_index=1,z_loc=1,z0=1,alpha=1,R=1,title='da',color='white',view_elev=0, view_azim=0,sub_fig=False,alpha_p=1,R_mat=np.eye(3),surf=False):
 
        ax_1= self.axes[str(plot_index)]
#        current_plot_num=int(str(self.row_num)+str(self.column_num)+str(plot_index))
 #       self.ax_1 = self.fig.add_subplot(current_plot_num, projection='3d')
        phi_t=np.linspace(0,2*np.pi,100)
        if surf==True:
            x=R*np.cos(phi_t)
            y=R*np.sin(phi_t)
        else:
            theta_0=np.arccos(z0/(R)+0.01)
            x=R*np.sin(theta_0)*np.cos(phi_t)
            y=R*np.sin(theta_0)*np.sin(phi_t)


        ax_1.plot3D(x,y,z0,color,zorder=10,linewidth=5,alpha=alpha_p)
        #p3dc.set_fc(colors)
        ax_1.view_init(view_elev, view_azim)
        ax_1.set_title(title, fontsize=20)

        if sub_fig:
            sub_axis=self.add_subplot_axes(ax_1)
            sub_axis.get_xaxis().set_visible(False)
            sub_axis.get_yaxis().set_visible(False)
            phi=np.linspace(0,2*np.pi,100)
            a=alpha*R
            c=R
            x=c*np.cos(phi)
            y=a*np.sin(phi)
            z=c*np.sin(phi)
 
            sub_axis.plot(x,y)
            sub_axis.plot(x,z,'--')
            sub_axis.plot(x,z0*np.ones(len(x)) )
            
            sub_axis.plot(x,((z0/R)*R*alpha)*np.ones(len(x)) ,'--')
            sub_axis.set_title('', fontsize=20)
            sub_axis.set_xlim(-a,a)
            sub_axis.set_ylim(-a,a)
        else:
            pass

        
    def plot_func_surface_flat(self,field_cood,Ct_id=0,plot_index=1,title='Top Layer Nematic Field',color_range=[0,1],set_range=True,view_elev=0, view_azim=0):
        mesh_cood=self.mesh_cood[Ct_id]
        ax_1= self.axes[str(plot_index)]
        self.fig.delaxes(ax_1)
        
        ax_1 = self.fig.add_subplot(self.row_num, self.column_num,plot_index) #,gridspec_kw={'width_ratios':[3,3], 'height_ratios':[3,3],'wspace':0.4,'hspace':0.4} 
#        current_plot_num=int(str(self.row_num)+str(self.column_num)+str(plot_index))
 #       self.ax_1 = self.fig.add_subplot(current_plot_num, projection='3d')
        self.axes={str(plot_index):ax_1}
        if set_range==True:
            p3dc=ax_1.scatter(mesh_cood[:,0],mesh_cood[:,1] , c=np.round(field_cood,2), s=100,  marker='^',cmap='Spectral',alpha=0.1,vmin=color_range[0], vmax=color_range[1],zorder=3 ) #, facecolors=cmap(colors), shade=True, alpha=1
            cl=self.fig.colorbar(p3dc, ax=ax_1,label='Order strength ')
            cl.mappable.set_clim(color_range[0],color_range[1])
        else:
            p3dc=ax_1.scatter(mesh_cood[:,0],mesh_cood[:,1] , c=np.round(field_cood,2), s=100,  marker='^',cmap='Spectral',alpha=1,zorder=3) #, facecolors=cmap(colors), shade=True, alpha=1
            cl=self.fig.colorbar(p3dc, ax=ax_1,label='Order strength ')
        #p3dc.set_fc(colors)
        #ax_1.view_init(view_elev, view_azim)
        ax_1.set_title(title, fontsize=20)

    def plot_quiver(self,_director,Ct_id=0,plot_index=1,plot_density=10,title='Top Layer Nematic Field',view_elev=0, view_azim=0,color='black',vector_length=1,vector_width=0.5):

        """
        start_ele_1=int(len(self.mesh_cood)*field_index_1)
        end_ele_1=int(len(self.mesh_cood)*(field_index_1+1))
        start_ele_2=int(len(self.mesh_cood)*field_index_2)
        end_ele_2=int(len(self.mesh_cood)*(field_index_2+1))
        start_ele_3=int(len(self.mesh_cood)*field_index_3)
        end_ele_3=int(len(self.mesh_cood)*(field_index_3+1))"""
        mesh_cood=self.mesh_cood[Ct_id]
        divide_plane_view=self.view_angle(Ct_id,view_elev, view_azim)

        _director_x=_director[0,:]#field_cood[start_ele_1:end_ele_1]
        _director_y=_director[1,:]#field_cood[start_ele_2:end_ele_2]
        _director_z=_director[2,:]#field_cood[start_ele_3:end_ele_3]
    
        #reduced_mesh=mesh_cood[:,0][dir_mesh]

        ax_1= self.axes[str(plot_index)]
        ax_1.view_init(view_elev, view_azim)
        p3dc=ax_1.quiver(mesh_cood[:,0][divide_plane_view][::plot_density],mesh_cood[:,1][divide_plane_view][::plot_density] ,mesh_cood[:,2][divide_plane_view][::plot_density], _director_x[divide_plane_view][::plot_density], _director_y[divide_plane_view][::plot_density], _director_z[divide_plane_view][::plot_density],pivot = 'middle', arrow_length_ratio=0,lw=vector_width,length=vector_length,color = color,zorder=11)
#        p3dc=ax_1.quiver(mesh_cood[:,0][::plot_density],mesh_cood[:,1][::plot_density] ,mesh_cood[:,2][::plot_density], _director_x[::plot_density], _director_y[::plot_density], _director_z[::plot_density],pivot = 'middle', arrow_length_ratio=0,lw=0.5,length=0.12,color = color,zorder=10)
        
        ax_1.set_title(title, fontsize=20)
       # self.fig.colorbar(p3dc, ax=ax_1,shrink=0.2, aspect=5,label='')
       

    def plot_quiver_flat(self,_director,Ct_id=0,plot_index=1,plot_density=10,title='Top Layer Nematic Field',view_elev=0, view_azim=0,color='black'):

        """
        start_ele_1=int(len(self.mesh_cood)*field_index_1)
        end_ele_1=int(len(self.mesh_cood)*(field_index_1+1))
        start_ele_2=int(len(self.mesh_cood)*field_index_2)
        end_ele_2=int(len(self.mesh_cood)*(field_index_2+1))
        start_ele_3=int(len(self.mesh_cood)*field_index_3)
        end_ele_3=int(len(self.mesh_cood)*(field_index_3+1))"""
        mesh_cood=self.mesh_cood[Ct_id]
        divide_plane_view=self.view_angle(Ct_id,view_elev, view_azim)

        _director_x=_director[0,:]#field_cood[start_ele_1:end_ele_1]
        _director_y=_director[1,:]#field_cood[start_ele_2:end_ele_2]
 
    
        #reduced_mesh=mesh_cood[:,0][dir_mesh]

        ax_1= self.axes[str(plot_index)]
 
        p3dc=ax_1.quiver(mesh_cood[:,0][::plot_density],mesh_cood[:,1][::plot_density], _director_x[::plot_density], _director_y[::plot_density],pivot = 'middle',headwidth=0 ,units='width'  )
#        p3dc=ax_1.quiver(self.mesh_cood[:,0][::plot_density],self.mesh_cood[:,1][::plot_density] ,self.mesh_cood[:,2][::plot_density], field1_reduced[::plot_density], field2_reduced[::plot_density], field3_reduced[::plot_density],pivot = 'middle', arrow_length_ratio=0,lw=0.5,length=0.3,color = 'black',normalize=True, zorder = 1)
        
        ax_1.set_title(title, fontsize=20)
       # self.fig.colorbar(p3dc, ax=ax_1,shrink=0.2, aspect=5,label='')
    def  chirality(self,_director,Ct_id=0,middle_width=0.3):
        mesh_cood=self.mesh_cood[Ct_id]
        z_mid_cord=np.abs(mesh_cood[:,2])<middle_width
        _director_x=_director[0,:][z_mid_cord]#field_cood[start_ele_1:end_ele_1]
        _director_y=_director[1,:][z_mid_cord]#field_cood[start_ele_2:end_ele_2]
        _director_z=_director[2,:][z_mid_cord]#field_cood[start_ele_3:end_ele_3]
        
        return np.sum(np.abs(_director_z))/len(_director_z)

    def save_fig(self,path_loc, index,dpi_num,name, transparent=False):
        self.path_loc=path_loc
        plt.gca(projection='3d').set_axis_off()
        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0)
        if transparent:
            plt.savefig(path_loc+'/'+name+'.png', dpi=dpi_num, transparent=True)
        else:
            plt.savefig(path_loc+'/'+name+'.png', dpi=dpi_num)
        plt.close()


    def overlay_img(self,top_img_path,bot_img_path,out_path,alpha=1, beta=1, gamma=0,blur_strength=8):


        background = Image.open(bot_img_path)
        overlay = Image.open(top_img_path)
        radius=2

        #background = background.convert("RGBA")
        #overlay = overlay.convert("RGBA")
        background = background.filter(ImageFilter.GaussianBlur(blur_strength))
        enhancer = ImageEnhance.Contrast(background)

        factor = 2 #gives original image
        background = enhancer.enhance(factor)
        background.paste(overlay, (0,0), mask = overlay)
        #new_img = Image.blend(background, overlay, 0)

        background.save(out_path,"png")
        #source1 = cv2.imread(bot_img_path, cv2.IMREAD_COLOR)
        #source2 = cv2.imread(top_img_path, cv2.IMREAD_COLOR)
        #dest=cv2.addWeighted(source1, alpha, source2, beta, gamma)
        #cv2.imwrite(out_path, dest)
        #cv2.destroyAllWindows()
    def export_csv(self,array_i,header_name='name',file_name='file.csv'):
        df = pd.DataFrame({header_name :array_i})
        df.to_csv(file_name, index=False)

    def close_fig(self):
        plt.close()
    def save_data(self,data,data_file_name,append_=True):
        if append_:
            with open(data_file_name, "a") as f:
                np.savetxt(f, data)
        else:
            with open(data_file_name, "w+") as f:
                np.savetxt(f, data)

    def video_out(self,N,movie_loc):
       # import cv2 
        img_array=[]
        for i in range(0,N):
            fig_loc=self.path_loc+'/test_'+str(i)+'.png'
            img = cv2.imread(fig_loc)
           # print(img.shape)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        out = cv2.VideoWriter(movie_loc,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
# %%
