import numpy as np
import matplotlib.pyplot as plt
import os
from mayavi.mlab import *
from mayavi import mlab
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

spr=10
dt=0.01   
dt_solver=0.002
t_start=0
t_stop=5000

model_dim=3
model_name='L63'
base_type='state_noisy'
num_clv=3
sigma=0.0
coord=['X','Y','Z']

base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=500.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))[10000:20000]

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63/Sensitivity'
os.chdir(data_path)

os.chdir('sigma={}_qr=1'.format(sigma))
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]

V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]  

# Cosines between the 1st and 2nd clv
cosines=np.zeros((base_traj.shape[0]))
for i in range(base_traj.shape[0]):
        cosines[i]=np.dot(V[i,:,0],V[i,:,1])

# Regime change in L63, by using CLVs
#plot the state and the trajectory

def shorten(arr, factor=1):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/factor)
    new_arr = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        new_arr[i] = arr[i*factor]
    return new_arr

os.chdir(plots_path)

clv_index=0
x,y,z=shorten(base_traj[:,0]),shorten(base_traj[:,1]),shorten(base_traj[:,2])
u,v,w=shorten(V[:,0,clv_index]),shorten(V[:,1,clv_index]),shorten(V[:,2,clv_index])

mlab.figure(bgcolor=(1, 1, 1))
my_field=mlab.points3d(x,y,z,cosines,colormap='CMRmap',scale_mode='none',opacity=0.5,scale_factor=0.5)
#my_field=mlab.quiver3d(x,y,z,u1,v1,w1,color=(0.1,0.5,0.1))

coord_lims=(-20,20,-30,30,0,50)
mlab.axes(my_field, color=(.7, .7, .7), extent=coord_lims, xlabel='x', ylabel='y',zlabel='z')
mlab.outline(my_field, color=(.3, .3, .3), extent=coord_lims)
mlab.text(0, 0, 'CLV {}'.format(clv_index+1), z=-4, width=0.5)
mlab.show()


