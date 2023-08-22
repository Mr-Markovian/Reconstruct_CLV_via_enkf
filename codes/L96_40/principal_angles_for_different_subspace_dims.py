"""We calculate the cosines of the angles for the assimialted trajectories with the observation
noise of the form mu times Identity."""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from scipy.linalg import subspace_angles

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

dt=0.05  
dt_solver=0.01
model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=11

spr=1
data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_{}_seed_{}/Sensitivity'.format(model_dim,seed_num)
os.chdir(data_path)

#os.mkdir('PA')

#Load the data for the actual trajectory
base_type='state_noisy'
os.chdir('sigma=0.0')
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]
os.chdir('..')

# We have first 10000 time steps as the transient to converge to BLVs.

#data_path='/home/shashank/Documents/Enkf_for_clv/data/L96/L96-40/assimilated_trajs_seed_3/ob1'
#os.chdir(data_path)
chosen_subspaces=np.array([2,5,10,15,20])
#for sigma in sigmas:
sigma=0.3
for subspace_num in chosen_subspaces:
    #folder_name='ebias=4.0_ecov=2.0_obs=20_ens=20_mu={}_gap=0.05_alpha=1.0_loc=gaspri_r=4'.format(mu)
    os.chdir(data_path+'/sigma={}'.format(sigma))
    #os.chdir(data_path+'/ob{}'.format(4)) 

    #os.chdir(data_path+'/mu={}_times_I20'.format(mu))
    base_type='state_noisy'
    C1=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    G1=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    
    # Principals of the two subspaces
    cosines3=np.zeros((G.shape[0],subspace_num))
    for i in range(G.shape[0]):
        D=subspace_angles(G1[i,:,:subspace_num],G[i,:,:subspace_num])
        cosines3[i]=np.rad2deg(np.flip(D))

    # save thes
    os.chdir(data_path+'/PA')
    np.save('blv_principal_cosine_sigma={}_model_dim={}_subspaces_={}.npy'.format(sigma,model_dim,subspace_num),cosines3)
