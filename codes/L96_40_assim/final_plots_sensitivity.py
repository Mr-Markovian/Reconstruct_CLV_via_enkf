import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
import os
import seaborn as sns
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
num_BLV=model_dim
num_clv=model_dim
startpt=0 # starting point of the forward transient
qrstep=5
seed_num=11

# Change to data path
data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_40_assim/ob3'
plots_path='/home/shashank/Documents/enkf_for_clv2/plots/L96_{}'.format(model_dim)

os.chdir(data_path)
rmse=np.load('rmse_versus_mu.npy')

mus=np.array([0.3,0.5,0.7,0.9,1.0])
bars=np.zeros((2,len(mus),num_BLV))
medians=np.zeros((len(mus),num_BLV))

os.chdir(data_path+'/CLV_versus_mu')
#plt.figure(figsize=(14,8))
fig,ax=plt.subplots(figsize=(14,8))
for i,mu in enumerate(mus):
    #clv_cosines=np.load('blv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
    # plt.plot(np.arange(1,num_clv+1),np.mean(clv_cosines,axis=0))
    # plt.errorbar(x=np.arange(1,num_clv+1),y=np.mean(clv_cosines,axis=0),yerr=np.std(clv_cosines,axis=0),capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20)
    clv_cosines=np.rad2deg(np.arccos(np.load('clv_cosine_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_clv))))
    #clv_cosines=np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
    quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
    asymmetric_bar=np.zeros((2,num_clv))
    asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
    asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
    #plt.plot(np.arange(1,num_clv+1),np.mean(blv_cosines,axis=0))
    ax.errorbar(x=np.arange(1,num_clv+1),y=np.median(clv_cosines,axis=0),yerr=asymmetric_bar,capsize=4,label=r'$\mu$={}'.format(mu),marker='.',ms=20,c=colors[i])

ax.set_xlabel(r'$i^{th}$ CLV',fontsize=25)
ax.set_ylim(0,90)
#plt.yticks(np.arange(0.3,1.1,0.1))
# ax2=ax.twiny()
# #ax2.set_xlim(ax.get_xlim())
# ax2.set_xticks(mus)
# ax2.set_xticklabels(np.round(rmse,2))
# ax2.set_xlabel('RMSE',fontsize=25)
ax.set_xticks(np.arange(1,num_clv+1,3))
ax.set_ylabel(r'Relative angle w.r.t true CLV$(\theta)$',fontsize=25)
#plt.ylabel(r'cos$(\theta)$')
plt.tight_layout()
ax.legend(loc='lower center',ncol=3,fontsize=20)
#plt.savefig('clv_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))
os.chdir(plots_path)
plt.savefig('CLV_angle_sensitivity_da_L96-{}_seed_{}.pdf'.format(model_dim,seed_num))

# ax2=ax.twiny()
# ax.set_xlabel(r'$\mu \to$',fontsize=25)
# ax.set_ylim(0.0,60)
# ax.set_xticks(mus)
# ax.set_ylabel(r'$\theta \to$',fontsize=25)
# #ax2.set_xticks(rmse)
# ax2.set_xlim(ax.get_xlim())
# ax2.set_xticks(mus)
# ax2.set_xticklabels(np.round(rmse,2))
# ax2.set_xlabel('RMSE',fontsize=25)
#plt.legend(loc='lower center',ncol=len(mus))
#plt.legend(loc='upper right')

# # calculate data for the inset
# mus1=np.array([0.1,0.2,0.3,0.4,0.5])
# bars1=np.zeros((2,len(mus1),num_BLV))
# medians1=np.zeros((len(mus1),num_BLV))

# for i,mu in enumerate(mus1):
#     BLV_cosines=np.rad2deg(np.arccos(np.load('BLV_cosine_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_BLV))))
#     quantiles_=np.quantile(BLV_cosines,[0.25,0.50,0.75],axis=0)
#     bars1[0,i,:]=quantiles_[1]-quantiles_[0]
#     bars1[1,i,:]=quantiles_[2]-quantiles_[1]
#     medians1[i]=np.median(BLV_cosines,axis=0)

# ax_inset = ax.inset_axes([0.05, 0.5, 0.6, 0.5])#)
# ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True))
# for i in range(num_BLV):     
#     ax_inset.errorbar(x=mus1,y=medians1[:,i],yerr=bars1[:,:,i],capsize=4,mec='black',label=r'$BLV$={}'.format(i+1),marker='.',ms=20,c=colors[i])

# ax_inset.set_xticks(mus1)
#ax_inset.set_ylabel('')
# plots_path='/home/shashank/Documents/enkf_for_clv2/plots/L96_40'
# os.chdir(plots_path)
# plt.savefig('CLV_angle_versus_mu_L96-{}_assim.pdf'.format(model_dim),bbox_inches='tight', pad_inches=0)


# for mu in mus:
#     BLV_cosines=np.rad2deg(np.arccos(np.load('BLV_cosine_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_BLV))))
#     # plt.plot(np.arange(1,num_BLV+1),np.mean(BLV_cosines,axis=0))
#     # plt.errorbar(x=np.arange(1,num_BLV+1),y=np.mean(BLV_cosines,axis=0),yerr=np.std(BLV_cosines,axis=0),capsize=4,label=r'$\mu$={}'.format(mu),marker='.',ms=20)

#     #BLV_cosines=np.rad2deg(np.arccos(np.load('blv_cosine_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_BLV))))
#     quantiles_=np.quantile(BLV_cosines,[0.25,0.50,0.75],axis=0)
#     asymmetric_bar=np.zeros((2,num_BLV))
#     asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
#     asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
#     #plt.plot(np.arange(1,num_BLV+1),np.mean(blv_cosines,axis=0))
#     plt.errorbar(x=np.arange(1,num_BLV+1),y=np.median(BLV_cosines,axis=0),yerr=asymmetric_bar,capsize=4,label=r'$\mu$={}'.format(mu),marker='.',ms=20)


# plt.xlabel(r'LV-index $\to$')
# #plt.yticks(np.arange(0.3,1.1,0.1))
# plt.xticks(np.arange(1,num_BLV+1))
# plt.ylabel(r'$\theta \to$')
# #plt.legend(loc='lower center',ncol=len(mus))
# plt.legend(loc='center right')
# plt.savefig('BLV_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))
# #plt.savefig('blv_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))

# plt.figure(figsize=(16,8))
# for mu in mus:
#     lexp=np.load('exp_mu={}_model_dim={}_num_cl={}.npy'.format(mu,model_dim,num_BLV))
#     plt.scatter(np.arange(1,num_BLV+1),lexp,label=r'$\mu$={}'.format(mu),s=20)

# plt.scatter(np.arange(1,num_BLV+1),lexp,label=r'$\mu$={}'.format(0.0),s=20,c='black')
# plt.xlabel(r'index $\to$')
# plt.ylabel(r'$\hat \lambda $')
# plt.xticks(np.arange(1,num_BLV+1))
# plt.legend()
# plt.savefig('exponent_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))


