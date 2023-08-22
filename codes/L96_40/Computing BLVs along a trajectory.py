"""Computing BLVs along a trajectory"""
from jax import jvp,vmap,jit,jacfwd
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import os
from ode_solvers import rk4_solver as solver
from testbed_models import L96
qr_decomp=jnp.linalg.qr

# Compute the Jacobian of the system from the model
model=L96
model_dim=20
num_clv=20

#jacobian of the model
jac_model=jit(jacfwd(model))

# When the number of clvs is less than the jacobian dimension, we do not compute the full jacobian, but use jvp 
@jit
def model_plus_tangent_propagator(X_):
    "model(x) + jacobian(x)V for state space and tanegent linear evolution at a point"
    x_eval=X_[:,0]  # Nx1
    in_tangents=X_[:,1:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp, jit(model) , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(1), out_axes=(None,1))((in_tangents,)) 
    Y_=X_.at[:,0].set(y)
    Z_=Y_.at[:,1:].set(out_tangents)
    return Z_

# Reorthonormalization step of the output tangent vectors is done using this function
@jit
def Output_tangent_QR(x_):
    """Split the vectors for the trajectory point and the tangents and store 
    the vectors and the upper triangular matrix Q and R respectively"""
    Q_,R_=qr_decomp(x_[:,1:])  # QR decomposition
    S=jnp.diag(jnp.sign(jnp.diag(R_)))
    temp_=x_.at[:,1:].set(Q_@S)
    return temp_,Q_@S,S@R_
    
# Load the trajectory of the system
dt=0.05
dt_solver=0.01
n_iters=int(dt/dt_solver)

# Change to data path
startpt=100 # starting point of the forward transient
qrstep=10

# Solver for three regions
n1=10000  # number of time steps in forward transient

base_type='State'
data_path='/home/shashank/Documents/enkf_for_clv2/data/L96_20/Convergence tests'
os.chdir(data_path)
#os.mkdir('L96_BLV_test1')
os.chdir('L96_BLV_test1')

base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=600.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))
print(base_traj.shape)

# orthogonal perturbations at t=0
X_start=np.zeros((model_dim,num_clv+1))
X_start[:,1:]=np.eye(model_dim)[:,:num_clv]  
X_start[:,0]=base_traj[0]

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function

my_solver=jit(partial(solver,rhs_function=model_plus_tangent_propagator,time_step=dt_solver))

#Store local growth rates, the G, R and C matrices
#local_growth_rates=np.zeros((n1-startpt,num_clv))
Store_G=np.zeros((int(n1/qrstep),model_dim,num_clv))
#Store_R=np.zeros((int(n1/qrstep),num_clv,num_clv))
#traj=np.zeros_like(base_traj)
#print(traj.shape)

for i in range(n1-1):
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop*1.0
    
    #Compute the exponents and store the matrices
    
    #local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    if(np.mod(i,qrstep)==qrstep-1):
        X_start,Store_G[int(i/qrstep)],_=Output_tangent_QR(X_stop)
    #    print(np.linalg.norm(X_start[:,0]-base_traj[i+1]))
    #local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    #traj[i+1]=X_start[:,0]
    X_start=X_start.at[:,0].set(base_traj[i+1])
    
    if (i==startpt):
        X_start=X_start.at[:,1:].set(jnp.eye(model_dim)[:,num_clv])


np.save('BLV_qr{}_p{}_mtp.npy'.format(qrstep,startpt),Store_G)
#np.save('exponents.npy',local_growth_rates)
print('Job done')

# plt.plot(np.arange(100),traj[:100]-base_traj[:100])
# plt.show()
