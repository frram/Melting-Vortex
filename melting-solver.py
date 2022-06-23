import numpy as np
import scipy.fftpack as sf
import math as mt
import glob

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

    dx = lx/nx

    wf = sf.fft2(w)
    psif = wf/(-(KX2+KY2)); psif[0,0] = 0
    uxf = -1.j*KY*psif; ux = np.real(sf.ifft2(uxf *KXD*KYD))
    uyf = 1.j*KX*psif; uy = np.real(sf.ifft2(uyf *KXD*KYD))
    obu = np.gradient(ux,axis=1)*np.gradient(uy,axis=0)-np.gradient(uy,axis=1)*np.gradient(ux,axis=0)
    obuf = sf.fft2(obu); obuf_c = np.conj(obuf)
    energy_k_tmp = obuf*obuf_c
    usq = ux**2 + uy**2
    energy = np.sum(np.sum(usq[:,:]*dx,axis=0)*dx)/lx**2

    whst = np.zeros((nt//isav,nx,ny))
    psihst = np.zeros((nt//isav,nx,ny))
    obuhst = np.zeros((nt//isav,nx,ny))
    wfhst = np.zeros((nt//isav,nx,ny))
    psifhst = np.zeros((nt//isav,nx,ny))
    energyfhst = np.zeros((nt//isav,nx,ny))
    energyhst = np.zeros((nt//isav))

    psifhst[0,:,:] = abs(np.fft.fftshift(psif))
    wfhst[0,:,:] = abs(np.fft.fftshift(wf))
    energyfhst[0,:,:] = abs(np.fft.fftshift(energy_k_tmp))

    whst[0,:,:] = np.real(sf.ifft2(wf))
    psihst[0,:,:] = np.real(sf.ifft2(psif))
    obuhst[0,:,:] = obu[:,:]
    energyhst[0] = energy

    for it in range(1,nt):
        gw1 = adv(wf,omega,alpha,Fw)
        gw2 = adv(wf+0.5*dt*gw1,omega,alpha,Fw)
        gw3 = adv(wf+0.5*dt*gw2,omega,alpha,Fw)
        gw4 = adv(wf+dt*gw3,omega,alpha,Fw)

        wf = wf+dt*(gw1+2*gw2+2*gw3+gw4)/6

        if(it%isav==0):
            psif = wf/(-(KX2+KY2)); psif[0,0]=0
            uxf = -1.j*KY*psif; ux = np.real(sf.ifft2(uxf *KXD*KYD))
            uyf = 1.j*KX*psif; uy = np.real(sf.ifft2(uyf *KXD*KYD))
            obu = np.gradient(ux,axis=1)*np.gradient(uy,axis=0)-np.gradient(uy,axis=1)*np.gradient(ux,axis=0)
            obuf = sf.fft2(obu); obuf_c = np.conj(obuf)
            energy_k_tmp = obuf*obuf_c
            usq = ux**2 + uy**2
            energy = np.sum(np.sum(usq*dx,axis=0)*dx)/lx**2
            w = np.real(sf.ifft2(wf))
            psi = np.real(sf.ifft2(psif))

            psifhst[it//isav,:,:] = abs(np.fft.fftshift(psif))
            wfhst[it//isav,:,:] = abs(np.fft.fftshift(wf))
            energyfhst[it//isav,:,:] = abs(np.fft.fftshift(energy_k_tmp))

            whst[it//isav,:,:] = w
            psihst[it//isav,:,:] = psi
            obuhst[it//isav,:,:] = obu[:,:]
            energyhst[it//isav] = energy
    return whst, wfhst, psihst, psifhst, obuhst, energyfhst, energyhst

def adv(wf,omega,alpha,Fw):
    psif = wf/(-(KX2+KY2)); psif[0,0]=0
    w = np.real(sf.ifft2(wf))

    wxf = 1.j*KX*wf; wx = np.real(sf.ifft2(wxf *KXD*KYD))
    wyf = 1.j*KY*wf; wy = np.real(sf.ifft2(wyf *KXD*KYD))
    etaf = -(KX2+KY2)*wf; eta = np.real(sf.ifft2(etaf))
    uxf = -1.j*KY*psif; ux = np.real(sf.ifft2(uxf *KXD*KYD))
    uyf = 1.j*KX*psif; uy = np.real(sf.ifft2(uyf *KXD*KYD))

    advf = -ux*wx - uy*wy + (1/omega)*eta - alpha*w + Fw

    advff = sf.fft2(advf)

    return advff

nx=128; ny=128; nt=1000000; isav=nt//25
dt=1e-2
lx=2*np.pi/0.15; ly=lx
dx=lx/nx; dy=ly/ny
x = np.arange(nx)*dx
y = np.arange(ny)*dy
X,Y=np.meshgrid(x,y)

n=4; k=10; nu=1e-2; alpha_p=0.55
Re=1.375 #SX state
omega=n*Re # increment by 0.5
Famp=(nu**2)*(k**3)*Re
alpha = n*nu*alpha_p*k/Famp

Fw = -1*n**3*(np.cos(n*X*0.15)+np.cos(n*Y*0.15))/omega

run_files = sorted(glob.glob('./melt-*.npz'))
run_iter = len(run_files)
if run_iter == 0:
    wnoise = []
    for v in range(1,3):
        for b in range(1,3):
            wn_temp = (np.sin(v*X*0.15+b*Y*0.15)+np.cos(v*X*0.15+b*Y*0.15))*(b**2/np.sqrt(v**2+b**2))
            wnoise.append(wn_temp)
    w = -1*n*(np.cos(n*X*0.15)+np.cos(n*Y*0.15))+0.0001*sum(wnoise)
else:
    data = np.load(run_files[-1])
    w_tmp = data['whst']
    w = w_tmp[-1,:,:]

whst, wfhst, psihst, psifhst, obuhst, energyfhst, energyhst = melting(nx,ny,lx,ly,nt,dt,alpha,w,Fw,omega,isav)

if run_iter < 10:
    np.savez('./melt-n'+str(n)+'-omega'+str(omega)+'-tmp_0'+str(run_iter)+'.npz',whst=whst,wfhst=wfhst,psihst=psihst,psifhst=psifhst,obuhst=obuhst,energyfhst=energyfhst,energyhst=energyhst,omega=omega)
else:
    np.savez('./melt-n'+str(n)+'-omega'+str(omega)+'-tmp_'+str(run_iter)+'.npz',whst=whst,wfhst=wfhst,psihst=psihst,psifhst=psifhst,obuhst=obuhst,energyfhst=energyfhst,energyhst=energyhst,omega=omega)