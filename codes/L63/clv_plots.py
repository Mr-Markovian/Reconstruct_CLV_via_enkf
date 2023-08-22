import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlow

colors = [colormap(0), colormap(100),colormap(200)]

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=20
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=25
mpl.rcParams['ytick.labelsize']=25
mpl.rcParams['axes.labelsize']=25

spr=3
model_dim=3
model_name='L63'
base_type='state_noisy'
num_clv=3
coord=['X','Y','Z']
dt=0.01
mu=0.0
sigma=0.0
t_start=0
t_stop=5000
dt_solver=0.002

start_idx=10000

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63'
plots_path='/home/shashank/Documents/enkf_for_clv2/plots/L63'
os.chdir(data_path)
base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=500.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))[10000:20000]

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63/Sensitivity'
os.chdir(data_path)

os.chdir('sigma={}_qr=1'.format(sigma))
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
base_traj=np.load('{}_g={}_sigma={}.npy'.format(base_type,dt,0.0))[start_idx+t_start:start_idx+t_stop]

#print(C.shape,G.shape,traj.shape)
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]

os.chdir(plots_path)

plot_pairs=[[0,1],[1,2],[0,2]]
for clv_index in range(num_clv):
    for l,m in plot_pairs:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(base_traj[:,l],base_traj[:,m],c=colors[1],markersize=1,alpha=0.4)
        ax.quiver(base_traj[::spr,l],base_traj[::spr,m],V[::spr,l,clv_index],V[::spr,m,clv_index],angles='xy',scale_units='xy',scale=1.0,color=colors[0],label='CLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
        ax.scatter(base_traj[0,l],base_traj[0,m],c='r',s=80,edgecolors='black')

        #ax.set_title('CLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
        ax.set_xlabel(r'${}\to$'.format(coord[l]))
        ax.set_ylabel(r'${}\to$'.format(coord[m]))

        if l==1:
                ax.get_yaxis().set_visible(False)

        #plt.legend()
        plt.tight_layout()
        plt.savefig('CLV{} in {}{}for_{}.png'.format(clv_index+1,coord[l],coord[m],'truth'),dpi=300)

print('Job done')

# Combinign plots 
# num_clv=1
# fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(16,8),sharey=True)
# fig.subplots_adjust(wspace=0, hspace=0)

# plot_pairs=[[0,2],[1,2]]
# for k,ax in enumerate(axes):
#     for clv_index in range(num_clv):
#         for l,m in plot_pairs:
#             ax.plot(base_traj[:,l],base_traj[:,m],c=colors[1],markersize=1,alpha=0.4)
#             ax.quiver(base_traj[::spr,l],base_traj[::spr,m],V[::spr,l,clv_index],V[::spr,m,clv_index],angles='xy',scale_units='xy',scale=1.0,color=colors[0],label='CLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
#             ax.scatter(base_traj[0,l],base_traj[0,m],c='r',s=80,edgecolors='black')
#             #ax.set_title('CLV{} in {}{} plane'.format(clv_index+1,coord[l],coord[m]))
#             ax.set_xlabel('{}-axis'.format(coord[l]))
#             if k==0:
#                 ax.set_ylabel('{}-axis'.format(coord[m]))
            
# plt.legend()
# plt.tight_layout()
# plt.savefig('CLV{} for {}{}for_{}.png'.format(1,coord[l],coord[m],'truth'))

# just check if the vectors are of magnitude 1
norms=np.sum(np.sum(V**2,axis=-1),axis=1)
