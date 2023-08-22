"""We plot the mean of cosines of the angles after the BLVs converge.
We also then plot the mean versus clv index for different values of sigma.
We also obtain the exponents and their values"""

import numpy as np
import os
import matplotlib.pyplot as plt

# mpl.rcParams['lines.markersize']=10
# mpl.rcParams['axes.titlesize']=30
# mpl.rcParams['legend.fontsize']=20
# mpl.rcParams['xtick.labelsize']=15
# mpl.rcParams['ytick.labelsize']=15
# mpl.rcParams['axes.labelsize']=20

dt=0.05  
dt_solver=0.01
model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=11
startpt=0 # starting point of the forward transient
qrstep=5


# Change to data path
data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_{}_seed_{}/Sensitivity'.format(model_dim,seed_num)
os.chdir(data_path)

#os.mkdir('CLV_versus_sigma')
#os.mkdir('BLV_versus_sigma')

#Load the data for the actual trajectory
base_type='state_noisy'
os.chdir('sigma=0.0')
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
os.chdir('..')

# We have first 10000 time steps as the transient to converge to BLVs.

sigmas=np.array([0.1])
#sigmas=np.array([1.0,2.0,3.0,4.0,5.0])
plt.figure(figsize=(14,8))

for sigma in sigmas:
    os.chdir(data_path+'/sigma={}'.format(sigma))
    C1=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
    #plt.scatter(np.absolute(C1[:,0,1]),np.absolute(C[:,0,1]),label='$\sigma$='.format(sigma))
    plt.plot(np.arange(C.shape[0]),C[:,0,1])
    plt.plot(np.arange(C.shape[0]),C1[:,0,1])

plt.show()



