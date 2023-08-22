import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from cmcrameri import cm
colormap=cm.batlow

colors = [colormap(0), colormap(150),colormap(200)]

data_path='/home/shashank/Documents/enkf_for_clv2/data/L63'
plots_path='/home/shashank/Documents/enkf_for_clv2/data/L63'

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=40
mpl.rcParams['legend.fontsize']=40
mpl.rcParams['xtick.labelsize']=40
mpl.rcParams['ytick.labelsize']=40
mpl.rcParams['axes.labelsize']=40

spr=3

model_dim=3
model_name='L63'
base_type1='State'
num_clv=3
coord=['X','Y','Z']
dt=0.01
dt_solver=0.002
t_start=0
t_stop=20000
start_idx=10000


# vectors for the true trajectory
os.chdir(data_path)
base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=500.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))

os.chdir(plots_path)

plt.figure(figsize=(16,10))
plt.plot(0.01*np.arange(t_stop-t_start),base_traj[t_start:t_stop,1],label='truth',c=colors[0])

plt.xlabel(r'$ time \to$')
plt.ylabel('{}'.format(coord[1]))
plt.legend()
plt.savefig('state_.pdf')
plt.show()
print('Job done')

