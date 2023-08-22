import jax.numpy as jnp
import numpy as np
def Halvorsen(x,a=1.4):
    return jnp.array([-a*x[0]-4*x[1]-4*x[2]-x[1]*x[1],-a*x[1]-4*x[2]-4*x[0]-x[2]*x[2],-a*x[2]-4*x[0]-4*x[1]-x[0]*x[0]])

def Rossler(x,a=0.2,b=0.2,c=18):
    return jnp.array([-x[1]-x[2],x[0]+a*x[1],b+x[2]*(x[0]-c)])

def Thomas(x,b=0.20):
    #0.18 
    return jnp.array([jnp.sin(x[1])-b*x[0],jnp.sin(x[2])-b*x[1],jnp.sin(x[0])-b*x[2]])

def Chua(x,alpha=15.395,beta=28.0,R=-1.143,C_2=-0.714):
    phi_x = C_2*x + 0.5*(R-C_2)*(jnp.abs(x+1)-jnp.abs(x-1))
    return jnp.array([alpha*(x[1]-x[0]-phi_x),x[0]-x[1]+x[2],-beta*x[1]])

# L96 2-level model, 
def L96_2level(t,x_,d_x,d_y,F_x,c=10,b=10,h=1):
    "2 scale lorenz 96 model, the input is combined vector x_=[x y] "
    X=x_[0:d_x]
    Y=x_[d_x:].reshape(d_x,d_y)
    dx_dt=-jnp.roll(X,-1)*(jnp.roll(X,-2)-jnp.roll(X,1))-X+F_x-(h*c/b)*jnp.sum(Y,axis=1)
    dy_dt=-c*b*(jnp.roll(Y,1,axis=1)*(jnp.roll(Y,2,axis=1)-jnp.roll(Y,-1,axis=1)))-c*Y+(h*c/b)*jnp.tile(X,(d_y,1)).T
    dx=jnp.zeros_like(x_)
    dx[0:d_x]=dx_dt
    dx[d_x:]=dy_dt.flatten()
    return dx

# L96 single level model with forcing=10
def L96(x_,forcing=8.):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    dx_dt=(jnp.roll(x_,-1)-jnp.roll(x_,2))*jnp.roll(x_,1)-x_+forcing
    return dx_dt

#Lorenz-63 with default values: sigma=10, rho=28, beta=8/3
def L63(x,sigma=10.,rho=28.,beta=8./3):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    return jnp.array([sigma*(x[1]-x[0]),x[0]*(rho-x[2])-x[1],x[0]*x[1]-beta*x[2]])
    

