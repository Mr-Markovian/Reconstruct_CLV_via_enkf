import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlowK

# A colormap in dark mode
#colormap=cm.vanimo

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63'
plots_path='/home/shashank/Documents/enkf_for_clv2/plots/L63'
mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

spr=20
dt=0.01   
sigma=0.0
dt_solver=0.002

model_dim=3
model_name='L63'
base_type='state_noisy'
num_clv=3
coord=['X','Y','Z']

base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=500.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))[10000:20000]

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63/Sensitivity'
os.chdir(data_path)

os.chdir('sigma={}_qr=1'.format(sigma))
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
print(np.mean(np.load('local_growth_rates_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type)),axis=0))


V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]  

# Cosines between the 1st and 2nd clv
cosines=np.zeros((base_traj.shape[0]))
for i in range(base_traj.shape[0]):
        cosines[i]=np.dot(V[i,:,0],V[i,:,1])

# Regime change in L63, by using CLVs
#plot the state and the trajectory

clv_index=0
os.chdir(plots_path)
fig = plt.figure(figsize=(16,10))
  
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
# plotting
my_scatter=ax.quiver(base_traj[::spr,0],base_traj[::spr,1],base_traj[::spr,2],V[::spr,0,clv_index],V[::spr,1,clv_index],V[::spr,2,clv_index],color='blue',label='truth')
#ax.plot(base_traj[:,0],base_traj[:,1],base_traj[:,2],c='grey',lw=1)
ax.set_xlabel(r'$X\to$')
ax.set_ylabel(r'$Y\to$')
ax.set_zlabel(r'$Z\to$')
ax.grid(False)
#ax.set_title('cosine between clv{} and clv{}'.format(1,2))
#fig.colorbar(my_scatter,shrink=0.5, aspect=5)
plt.legend()
plt.show()
#plt.savefig('clv_{}_3d_plot.png'.format(clv_index),dpi=300, bbox_inches='tight', pad_inches=0)