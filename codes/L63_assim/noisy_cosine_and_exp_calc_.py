"""We plot the mean of cosines of the angles after the BLVs converge.
We also then plot the mean versus clv index for different values of mu.
We also obtain the exponents and their values"""

import numpy as np
import os

# mpl.rcParams['lines.markersize']=10
# mpl.rcParams['axes.titlesize']=30
# mpl.rcParams['legend.fontsize']=20
# mpl.rcParams['xtick.labelsize']=15
# mpl.rcParams['ytick.labelsize']=15
# mpl.rcParams['axes.labelsize']=20

dt=0.01  
dt_solver=0.002
model_dim=3
model_name='L63'
num_clv=model_dim
seed_num=3
startpt=0 # starting point of the forward transient
qrstep=5


# Change to data path
data_path='/home/shashank/Documents/enkf_for_clv2/data/L63_assim/ob5'
os.chdir(data_path)
base_type='state'

#Load the data for the actual trajectory
os.chdir('state')
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]
os.chdir('..')

# We have first 10000 time steps as the transient to converge to BLVs.
base_type='analysis'

mus=np.array([0.1,0.3,0.5,0.7,0.9])

for mu in mus:
    os.chdir(data_path+'/ebias=6.0_ecov=2.0_obs=1_ens=25_mu={}_gap=0.01_alpha=1.0_loc=none_r=0'.format(mu))
    C1=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    G1=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    V1=np.zeros_like(G1)
    for i in range(G1.shape[0]):
        V1[i]=G1[i]@C1[i]

    # Angle between BLVs
    cosines=np.zeros((G.shape[0],num_clv))
    for i in range(G.shape[0]):
        for j in range(num_clv):
            cosines[i,j]=np.absolute(np.dot(G1[i,:,j],G[i,:,j]))

    # Angle between CLVs
    cosines2=np.zeros((G.shape[0],num_clv))
    for i in range(G.shape[0]):
        for j in range(num_clv):
            cosines2[i,j]=np.absolute(np.dot(V1[i,:,j],V[i,:,j]))

    # save these 
    local_growth_rate=np.load('local_growth_rates_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    store_lexp=np.mean(local_growth_rate,axis=0)
    os.chdir(data_path+'/BLV_versus_mu')
    np.save('exp_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_clv),store_lexp)
    np.save('blv_cosine_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_clv),cosines)

    os.chdir(data_path+'/CLV_versus_mu')
    np.save('clv_cosine_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_clv),cosines2)
    


