from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlowK

colors = [colormap(i*50) for i in range(5)]

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=25
mpl.rcParams['legend.fontsize']=25
mpl.rcParams['xtick.labelsize']=25
mpl.rcParams['ytick.labelsize']=25
mpl.rcParams['axes.labelsize']=25

dt=0.05   
dt_solver=0.01
model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=11
base_type='state_noisy'

data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_{}_seed_{}/Sensitivity'.format(model_dim,seed_num)
plots_path='/home/shashank/Documents/enkf_for_clv2/plots/L96_{}'.format(model_dim)

sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
os.chdir(data_path+'/sigma={}'.format(0.0))
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))
V=np.zeros_like(G)

for i in range(G.shape[0]):
    V[i]=G[i]@C[i]

plt.matshow(V[0])
plt.matshow(G[0])

plt.show()
#plt.xlabel(r'$i^{th}$ CLV',fontsize=25)
#plt.ylabel(r'Relative angle w.r.t true CLV$(\theta)$',fontsize=25)

#os.chdir(plots_path)
#plt.savefig('clv_coefficient_plot_L96-{}_seed_{}.pdf'.format(model_dim,seed_num))
#plt.savefig('blv_coefficient_plot_L96-{}_seed_{}.pdf'.format(model_dim,seed_num))


# plt.figure(figsize=(16,8))
# for sigma in sigmas:
#     lexp=np.load('exp_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
#     plt.scatter(np.arange(1,num_clv+1),lexp,label=r'$\sigma$={}'.format(sigma),s=20)

# plt.scatter(np.arange(1,num_clv+1),lexp,label=r'$\sigma$={}'.format(0.0),s=20,c='black')
# plt.xlabel(r'index $\to$')
# plt.ylabel(r'$\hat \lambda $')
# plt.xticks(np.arange(1,num_clv+1))
# plt.legend()
# plt.savefig('exponent_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))


