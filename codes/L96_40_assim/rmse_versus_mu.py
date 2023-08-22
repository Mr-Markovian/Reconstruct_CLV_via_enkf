import numpy as np
import os
import matplotlib.pyplot as plt

mus=np.array([0.3,0.5,0.7,0.9,1.0])

data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_40_assim/ob3'
os.chdir(data_path)

# true trajectory
os.chdir('state')
base_traj=np.load('state.npy')[1000:]
os.chdir('..')

# The rmse array
rmse=np.zeros_like(mus)

for i,mu in enumerate(mus):
    os.chdir(data_path+'/ebias=-5.0_ecov=1.0_obs=20_ens=25_mu={}_gap=0.05_alpha=1.0_loc=gaspri_r=4'.format(mu))
    analysis_traj=np.load('filtered_mean.npy')[1000:]
    rmse[i]=np.sqrt(np.mean(np.sum((analysis_traj-base_traj)**2,axis=1),axis=0))

plt.figure()
plt.scatter(mus,rmse/np.sqrt(40))
plt.xlabel(r'$\mu$',fontsize=20)
plt.ylabel(r'$RMSE$',fontsize=20)

os.chdir(data_path)
plt.savefig('scaled_rmse_versus_mu.pdf')
plt.show()
np.save('Scaled_rmse_versus_mu.npy',rmse)
