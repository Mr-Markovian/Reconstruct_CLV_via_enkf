"""Computing BLVs along a trajectory"""
from jax import jit
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import os
from ode_solvers import rk4_solver as solver
from clv_library_functions import model_plus_tangent_propagator,Output_tangent_QR
from testbed_models import L96

# Compute the Jacobian of the system from the model
model_dim=40
num_clv=40


# Load the trajectory of the system
dt=0.05
dt_solver=0.01
n_iters=int(dt/dt_solver)

# Change to data path
startpt=200 # starting point of the forward transient
qrstep=10

# Solver for three regions
n1=10000  # number of time steps in forward transient

base_type='State'
data_path='/home/shashank/Documents/enkf_for_clv2/data/L96-40/Convergence tests'

os.chdir(data_path)
os.chdir('L96_BLV_test6')

base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=500.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))
print(base_traj.shape)

# orthogonal perturbations at t=0
X_start=np.zeros((model_dim,num_clv+1))
X_start[:,1:]=np.eye(model_dim)[:,:num_clv]  
X_start[:,0]=base_traj[0]

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function

my_solver=jit(partial(solver,rhs_function=model_plus_tangent_propagator(model=jit(L96)),time_step=dt_solver))

Store_G=np.zeros((int(n1/qrstep),model_dim,num_clv))
#Store_R=np.zeros((int(n1/qrstep),num_clv,num_clv))

for i in range(n1-1):
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop*1.0
    
    #Compute the exponents and store the matrices
    
    #local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    if(np.mod(i,qrstep)==qrstep-1):
        X_start,Store_G[int(i/qrstep)],_=Output_tangent_QR(X_stop)
    #    print(np.linalg.norm(X_start[:,0]-base_traj[i+1]))
    #traj[i+1]=X_start[:,0]
    X_start=X_start.at[:,0].set(base_traj[i+1])
    
    if (i==startpt):
        X_start=X_start.at[:,1:].set(jnp.eye(model_dim)[:,num_clv])

#os.mkdir('L96_BLV_test1')
np.save('BLV_qr{}_p{}_mtp.npy'.format(qrstep,startpt),Store_G)
#np.save('exponents.npy',local_growth_rates)
print('Job done')

