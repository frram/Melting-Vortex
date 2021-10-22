import numpy as np
import scipy.fftpack as sf
import math as mt

def melting(nx,ny,lx,ly,nt,dt,alpha,w,Fw,omega,isav):
    global KX,KY,KX2,KY2,KXD,KYD
    kx = 2*np.pi/lx*np.r_[np.arange(nx/2),np.arange(-nx/2,0)]
    ky =2*np.pi/ly*np.r_[np.arange(ny/2),np.arange(-ny/2,0)]
    kxd=np.r_[np.ones(nx//3),np.zeros(nx//3+nx%3),np.ones(nx//3)]   #for de-aliasing
    kyd=np.r_[np.ones(ny//3),np.zeros(ny//3+ny%3),np.ones(ny//3)]   #for de-aliasing
    kx2=kx**2; ky2=ky**2
    KX ,KY =np.meshgrid(kx ,ky )
    KX2,KY2=np.meshgrid(kx2,ky2)
    KXD,KYD=np.meshgrid(kxd,kyd)

    wf = sf.fft2(w)
    psif = wf/(-(KX2+KY2)); psif[0,0] = 0

    whst = np.zeros((nt//isav,nx,ny))
    psihst = np.zeros((nt//isav,nx,ny))
    wfhst = np.zeros((nt//isav,nx,ny))
    wfhst[0,:,:] = abs(np.fft.fftshift(wf))
    whst[0,:,:] = np.real(sf.ifft2(wf))
    psihst[0,:,:] = np.real(sf.ifft2(psif))

    for it in range(1,nt):
        gw1 = adv(wf,omega,alpha,Fw)
        gw2 = adv(wf+0.5*dt*gw1,omega,alpha,Fw)
        gw3 = adv(wf+0.5*dt*gw2,omega,alpha,Fw)
        gw4 = adv(wf+dt*gw3,omega,alpha,Fw)

        wf = wf+dt*(gw1+2*gw2+2*gw3+gw4)/6

        if(it%isav==0):
            psif = wf/(-(KX2+KY2)); psif[0,0]=0

            w = np.real(sf.ifft2(wf))
            psi = np.real(sf.ifft2(psif))
            wfhst[it//isav,:,:] = abs(np.fft.fftshift(wf))
            whst[it//isav,:,:] = w
            psihst[it//isav,:,:] = psi
    return whst, wfhst, psihst

def adv(wf,omega,alpha,Fw):
    psif = wf/(-(KX2+KY2)); psif[0,0]=0

    # psi = np.real(sf.ifft2(psif))
    w = np.real(sf.ifft2(wf))

    wxf = 1.j*KX*wf; wx = np.real(sf.ifft2(wxf *KXD*KYD))
    wyf = 1.j*KY*wf; wy = np.real(sf.ifft2(wyf *KXD*KYD))
    etaf = -(KX2+KY2)*wf; eta = np.real(sf.ifft2(etaf))
    uxf = 1.j*KY*psif; ux = np.real(sf.ifft2(uxf *KXD*KYD))
    uyf = 1.j*KX*psif; uy = np.real(sf.ifft2(uyf *KXD*KYD))

    advf = ux*wx - uy*wy + (1/omega)*eta - alpha*w + Fw

    advff = sf.fft2(advf)

    return advff

nx=128; ny=128; nt=20000; isav=nt//10
alpha=1; omega=10
dt=1e-2
lx=2*np.pi/0.15; ly=lx
dx=lx/nx; dy=ly/ny
x = np.arange(nx)*dx
y = np.arange(ny)*dy
X,Y=np.meshgrid(x,y)

n=4
Fw = -1*n**3*(np.cos(n*X*0.15)+np.cos(n*Y*0.15))/omega
wnoise = []
for v in range(1,3):
    for b in range(1,3):
        wn_temp = (np.sin(v*X*0.15+b*Y*0.15)+np.cos(v*X*0.15+b*Y*0.15))*(b**2/np.sqrt(v**2+b**2))
        wnoise.append(wn_temp)
w = -1*n*(np.cos(n*X*0.15)+np.cos(n*Y*0.15))+0.0001*sum(wnoise)

whst, wfhst, psihst = melting(nx,ny,lx,ly,nt,dt,alpha,w,Fw,omega,isav)

np.savez('./melt-res'+str(nx)+'-n'+str(n)+'-alpha'+str(alpha)+'-omega'+str(omega)+'.npz',whst=whst, wfhst=wfhst, psihst=psihst)