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
num_clv=5
seed_num=11

data_path='/mnt/d/PhD Work/enkf_for_clv2/data/L96_{}_seed_{}/Sensitivity3'.format(model_dim,seed_num)
plots_path='/mnt/d/PhD Work/enkf_for_clv2/plots/L96_{}'.format(model_dim)

os.chdir(data_path+'/CLV_versus_sigma')
sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
plt.figure(figsize=(8,8))
for i,sigma in enumerate(sigmas):
    #clv_cosines=np.load('blv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
    # plt.plot(np.arange(1,num_clv+1),np.mean(clv_cosines,axis=0))
    # plt.errorbar(x=np.arange(1,num_clv+1),y=np.mean(clv_cosines,axis=0),yerr=np.std(clv_cosines,axis=0),capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20)
    clv_cosines=np.rad2deg(np.arccos(np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))))
    #clv_cosines=np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
    quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
    asymmetric_bar=np.zeros((2,num_clv))
    asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
    asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
    #plt.plot(np.arange(1,num_clv+1),np.mean(blv_cosines,axis=0))
    plt.errorbar(x=np.arange(1,num_clv+1),y=np.median(clv_cosines,axis=0),yerr=asymmetric_bar,capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20,c=colors[i])

plt.xlabel(r'$i^{th}$ CLV',fontsize=25)
plt.ylim(0,90)
#plt.yticks(np.arange(0.3,1.1,0.1))
plt.xticks(np.arange(1,num_clv+1,3))
plt.ylabel(r'Relative angle w.r.t true CLV$(\theta)$',fontsize=25)
#plt.ylabel(r'cos$(\theta)$')
plt.tight_layout()
plt.legend(loc='lower center',ncol=3,fontsize=20)
#plt.savefig('clv_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))
os.chdir(plots_path)
plt.savefig('5_out of 40_clv_angle_sensitivity_L96-{}_seed_{}.pdf'.format(model_dim,seed_num))

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


