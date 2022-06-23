import numpy as np
import random 
import glob 

rootdir = './'

sav_files = glob.glob(rootdir+'pst-n-pe-*.npz')
run_iter = len(sav_files)

# Grid size and time
nx, ny = 128, 128 # 128, 128 or 256, 256
nt = 64000 # adjust as needed
isav = nt//10
Lx, Ly = 2.0*np.pi, 2.0*np.pi
x = np.linspace(0,Lx,nx); y = x
dx, dy = Lx/(nx-1), Ly/(ny-1)
X, Y = np.meshgrid(x,y)

# Peclet number calculation
u = 0.5; d = np.pi/3; beta = 1
D = 1.7e-2
Pe = (u*d*beta)/D

# Flowlines 
m = 0 # shear mode
eta = 0 # shear strength
# psi_shear = (m/12)*np.sin(m*X/2)
mfile = sorted(glob.glob('./stream-*.npz'))
vel = np.load(mfile[-1])
ux = vel['ux']; uy = vel['uy']
w = ux
v = uy

# Noise
kappa = 0 # noise strength

# Initiliaze scalar array
n = np.zeros((ny,nx))
nhst = np.zeros((nt//isav,ny,nx))
nhst[0,:,:] = n

# Control parameters
sigma = 0.0009 # Need to figure this out still...
dt = sigma * dx * dy / D
alpha = dt/(2*dx)
gamma = (D* dt) / (dx**2)
g = -1; h = -1

# Finite difference solver
def passive(n,nhst,w,v,alpha,gamma,nx,ny,kappa,g,h,nt,isav):
    for k in range(0,nt-1,1):
        n[0,0] = 2*gamma*n[0,1]+ n[1,0]*(gamma-alpha*v[0,0]) + n[ny-2,0]*(gamma+alpha*v[0,0]) + n[0,0]*(1-4*gamma)-2*dx*(alpha*w[0,0]+gamma)*(g+kappa*random.randint(-100,100)/1000)
        for j in range(1,ny-1):
            n[j,0] = 2*gamma*n[j,1]+ n[j+1,0]*(gamma-alpha*v[j,0]) + n[j-1,0]*(gamma+alpha*v[j,0]) + n[j,0]*(1-4*gamma)-2*dx*(alpha*w[j,0]+gamma)*(g+kappa*random.randint(-100,100)/1000)
        for i in range(1,nx-1):
            n[0,i] = -alpha*(w[0,i]*(n[0,i+1]-n[0,i-1])+v[0,i]*(n[1,i]-n[ny-2,i])) + gamma*(n[0,i+1]+n[0,i-1]+n[1,i]+n[ny-2,i]-4*n[0,i]) + n[0,i]
        n[-1,:] = n[0,:]
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                n[j,i] = -alpha*(w[j,i]*(n[j,i+1]-n[j,i-1])+v[j,i]*(n[j+1,i]-n[j-1,i])) + gamma*(n[j,i+1]+n[j,i-1] + n[j-1,i] + n[j+1,i]-4*n[j,i]) +n[j,i]
        n[0,nx-1] = 2*gamma*n[0,nx-2]+ n[1,nx-1]*(gamma-alpha*v[0,nx-1]) + n[ny-2,nx-1]*(gamma+alpha*v[0,nx-1]) + n[0,nx-1]*(1-4*gamma)-2*dx*(alpha*w[0,nx-1]-gamma)*h*n[0,nx-1]
        n[ny-1,nx-1] = n[0,nx-1]
        for j in range(1,ny-1):
            n[j,nx-1] = 2*gamma*n[j,nx-2]+ n[j+1,nx-1]*(gamma-alpha*v[j,nx-1]) + n[j-1,nx-1]*(gamma+alpha*v[j,nx-1]) + n[j,nx-1]*(1-4*gamma)-2*dx*(alpha*w[j,nx-1]-gamma)*h*n[j,nx-1]
        if(k%isav == 0):
            nhst[k//isav,:,:] = n[:,:]
    return nhst

if run_iter == 0:
    nhst = passive(n,nhst,w,v,alpha,gamma,nx,ny,kappa,g,h,nt,isav)
    steps = nt
else:
    data = np.load(sav_files[-1])
    nsav = data['nhst']
    try:
        steps_tmp = data['steps']
    except KeyError:
        steps_tmp = 0
    n = nsav[-1,:,:]
    nhst[0,:,:] = n 
    nhst = passive(n,nhst,w,v,alpha,gamma,nx,ny,kappa,g,h,nt,isav)
    steps = steps_tmp + nt

if Pe < 10:
    if run_iter < 10:
        np.savez(rootdir+'pst-n-pe-000'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-0'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)
    else: 
        np.savez(rootdir+'pst-n-pe-000'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)
elif 100 > Pe >= 10:
    if run_iter < 10:
        np.savez(rootdir+'pst-n-pe-00'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-0'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)
    else:
        np.savez(rootdir+'pst-n-pe-00'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)
elif 1000 > Pe >= 100:
    if run_iter < 10:
        np.savez(rootdir+'pst-n-pe-0'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-0'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)
    else:
        np.savez(rootdir+'pst-n-pe-0'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)
else:
    if run_iter < 10:
        np.savez(rootdir+'pst-n-pe-'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-0'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)
    else:
        np.savez(rootdir+'pst-n-pe-'+str(int(Pe))+'-m-'+str(m)+'-alpha-'+str(eta)+'-kappa-'+str(kappa)+'-'+str(run_iter)+'.npz',nhst=nhst,w=w,v=v,nx=nx,dx=dx,nt=nt,steps=steps,dt=dt,D=D,m=m,eta=eta,kappa=kappa)

print('File save complete. Stored in: ' + rootdir)
print('Gridsize: (' + str(nx) + ',' +str(ny) + ')')
print('Time steps: ' + str(nt))
print('Size: ' + str(Lx))
print('iteration: ' + str(run_iter))
print('Diffusion: ' + str(D))
print('Total Steps: ' + str(int(steps)))
navgy = np.sum(nhst[-1,:,:]*dy,axis=0)/Ly
ntot = np.sum(navgy*dy)*Ly
print('Concentration: ' + str(ntot))