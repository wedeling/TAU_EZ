"""
*************************
* S U B R O U T I N E S *
*************************
"""

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, P):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    w_x_n = np.fft.irfft2(kx*w_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    w_y_n = np.fft.irfft2(ky*w_hat_n)
    
    VgradW_n = u_n*w_x_n + v_n*w_y_n
    
    #return to spectral space
    VgradW_hat_n = np.fft.rfft2(VgradW_n)
    
    VgradW_hat_n *= P
    
    return VgradW_hat_n

#compute the Jacobian of the smoothing PDE
def compute_VgradEF_hat(w_hat_n, EF_hat_n):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    EF_x_n = np.fft.irfft2(kx*EF_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    EF_y_n = np.fft.irfft2(ky*EF_hat_n)
    
    VgradEF_n = u_n*EF_x_n + v_n*EF_y_n
    
    #return to spectral space
    VgradEF_hat_n = np.fft.rfft2(VgradEF_n)
    
    VgradEF_hat_n *= P_LF
    
    return VgradEF_hat_n

#pseudo-spectral technique to solve for Fourier coefs of BCD components
def compute_MN_hat(w_hat_HF, w_hat_LF):
    
    #compute streamfunctions
    psi_hat_HF = w_hat_HF/k_squared_no_zero
    psi_hat_HF[0,0] = 0.0
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    
    #compute full and projected velocities
    u_HF = np.fft.irfft2(-ky*psi_hat_HF)
    u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    v_HF = np.fft.irfft2(kx*psi_hat_HF)
    v_LF = np.fft.irfft2(kx*psi_hat_LF)
    
    """
    #compute subgrid velocities
    du = u_HF - u_LF
    dv = v_HF - v_LF
    
    #return resolved part of the RST components (\bar{u_iu_j})
    M_hat = P_LF*np.fft.rfft2(u_LF*du - v_LF*dv + 0.5*(du*du - dv*dv))
    N_hat = P_LF*np.fft.rfft2(u_LF*dv + v_LF*du + du*dv)
    """
    
    M_HF = 0.5*(u_HF**2 - v_HF**2)
    M_LF = 0.5*(u_LF**2 - v_LF**2)
    N_HF = u_HF*v_HF
    N_LF = u_LF*v_LF
    
    dM_hat = P_LF*np.fft.rfft2(M_HF - M_LF)
    dN_hat = P_LF*np.fft.rfft2(N_HF - N_LF)
    M_LF_hat = P_LF*np.fft.rfft2(M_LF)
    N_LF_hat = P_LF*np.fft.rfft2(N_LF)
    M_HF_hat = P_LF*np.fft.rfft2(M_HF)
    N_HF_hat = P_LF*np.fft.rfft2(N_HF)
    
    return dM_hat, dN_hat, M_LF_hat, N_LF_hat, M_HF_hat, N_HF_hat

#pseudo-spectral technique to solve for Fourier coefs of RST components
def compute_rst_hat(w_hat):
    
    #compute streamfunction
    psi_hat = w_hat/k_squared_no_zero
    psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    u = np.fft.irfft2(-ky*psi_hat)
    u_bar = np.fft.irfft2(-P_LF*ky*psi_hat)
    v = np.fft.irfft2(kx*psi_hat)
    v_bar = np.fft.irfft2(P_LF*kx*psi_hat)
    
    #compute subgrid velocities
    u_prime = u - u_bar
    v_prime = v - v_bar
    
    #return resolved part of the RST components (\bar{u_iu_j})
    uu_hat = P_LF*np.fft.rfft2(u_prime*u_prime)
    uv_hat = P_LF*np.fft.rfft2(u_prime*v_prime)
    vv_hat = P_LF*np.fft.rfft2(v_prime*v_prime)

    return uu_hat, uv_hat, vv_hat

#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(w_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, P)
    
    #solve for next time step according to AB/BDI2 scheme
    w_hat_np1 = norm_factor*P*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                               2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    
    return w_hat_np1, VgradW_hat_n

#compute spectral filter
def get_P(cutoff):
    
    P = np.ones([N, N/2+1])
    
    for i in range(N):
        for j in range(N/2+1):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5():
  
    fname = HOME + '/samples/' + store_ID + '_t_' + str(np.around(t_end/day, 1)) + '.hdf5'
    
    print 'Storing samples in ', fname
    
    if os.path.exists(HOME + '/samples') == False:
        os.makedirs(HOME + '/samples')
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for q in QoI:
        h5f.create_dataset(q, data = samples[q])
        
    h5f.close()    

def draw_w():
    plt.subplot(111, xlabel=r'$t\;[days]$', ylabel=r'$\rho\left(r_{i+1}, \mathcal{C}_i\right)$')

    for c in range(C):
        plt.plot(T, rho[c])
        #plt.plot(test)

    plt.tight_layout()

def draw_2w():
    plt.subplot(121, xlabel=r'$t\;[days]$')
    plt.plot(T, DE, label=r'$\Delta_E$')
    plt.plot(T, R, label=r'$\widetilde{\Delta}_E$')
    plt.legend(loc=0)
    plt.subplot(122)
    plt.hist([DE, R], 20, label=[r'$\Delta_E$', r'$\widetilde{\Delta}_E$'])
    plt.legend(loc='upper right')
    plt.tight_layout()

def draw_stats():
    plt.subplot(121, xlabel=r't')
    plt.plot(T, energy_HF, label=r'$E^{HF}$')
    plt.plot(T, energy_LF, label=r'$E^{LF}$')
    plt.plot(T, energy_UP, label=r'$E^{UP}$')
    plt.legend(loc=0)
    plt.subplot(122, xlabel=r't')
    plt.plot(T, enstrophy_HF, label=r'$Z^{HF}$')
    plt.plot(T, enstrophy_LF, label=r'$Z^{LF}$')
    plt.plot(T, enstrophy_UP, label=r'$Z^{UP}$')
    plt.legend(loc=0)
    plt.tight_layout()

def draw_3w():
    plt.subplot(131, aspect='equal', title=r'$Q_1\; ' + r't = '+ str(np.around(t/day,2)) + '\;[days]$')
    plt.contourf(x, y, w_np1_HF, 100)
    plt.subplot(132, aspect='equal', title=r'$Q_2$')
    plt.contourf(x, y, w_np1_LF, 100)    
    plt.subplot(133, aspect='equal', title=r'$Q_3$')
    plt.contourf(x, y, dPsi_n, 100) 
    plt.tight_layout()

#compute the spatial correlation coeffient at a given time
def spatial_corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))

def get_psi_hat(w_hat_n):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0

    return psi_hat_n

###############################
# DATA-DRIVEN TAU SUBROUTINES #
###############################

def get_data_driven_tau(w_hat_n_LF, w_hat_n_HF, P, tau_max):
    
    E_HF = compute_E(P*w_hat_n_HF)
    E_LF = compute_E(w_hat_n_LF)
    
    dE = (E_HF - E_LF)/E_LF

    tau = tau_max*np.tanh(dE)
    
    return tau

def get_data_driven_tau_Z(w_hat_n_LF, w_hat_n_HF, P, tau_max):
    
    Z_HF = compute_Z(P*w_hat_n_HF)
    Z_LF = compute_Z(w_hat_n_LF)
    
    dZ = (Z_HF - Z_LF)/Z_LF

    tau = tau_max*np.tanh(dZ)
    
    return tau

def get_data_driven_tau_src_EZ(w_hat_n_LF, w_hat_n_HF, P, tau_max_E, tau_max_Z):
    
    E_LF, Z_LF, S_LF = get_EZS(w_hat_n_LF)

    src_E = E_LF**2/Z_LF - S_LF
    src_Z = -E_LF**2/S_LF + Z_LF

    E_HF = compute_E(P*w_hat_n_HF)
    Z_HF = compute_Z(P*w_hat_n_HF)

    dE = (E_HF - E_LF)/E_LF
    dZ = (Z_HF - Z_LF)/Z_LF

    tau_E = tau_max_E*np.tanh(dE)*np.sign(src_E)
    tau_Z = tau_max_Z*np.tanh(dZ)*np.sign(src_Z)
    
    return tau_E, tau_Z, dE, dZ

def get_EZS(w_hat_n):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)
    
    e_n = -0.5*psi_n*w_n
    z_n = 0.5*w_n**2
    s_n = 0.5*psi_n**2 

    E = simps(simps(e_n, axis), axis)/(2*np.pi)**2
    Z = simps(simps(z_n, axis), axis)/(2*np.pi)**2
    S = simps(simps(s_n, axis), axis)/(2*np.pi)**2

    return E, Z, S

#######################
# ORTHOGONAL PATTERNS #
#######################

def get_psi_hat_prime(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.

    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)

    nom = simps(simps(w_n*psi_n, axis), axis)
    denom = simps(simps(w_n*w_n, axis), axis)

    return psi_hat_n - nom/denom*w_hat_n

def get_w_hat_prime(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0, 0] = 0.

    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)

    nom = simps(simps(w_n*psi_n, axis), axis)
    denom = simps(simps(psi_n*psi_n, axis), axis)

    return w_hat_n - nom/denom*psi_hat_n

#compute the energy and enstrophy at t_n
def compute_E(w_hat_n):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)
    
    e_n = -0.5*psi_n*w_n

    E = simps(simps(e_n, axis), axis)/(2*np.pi)**2
    
    return E

#compute the energy and enstrophy at t_n
def compute_Z(w_hat_n):
    
    w_n = np.fft.irfft2(w_hat_n)
    
    z_n = 0.5*w_n**2

    Z = simps(simps(z_n, axis), axis)/(2*np.pi)**2
    
    return Z

#using the full dE ODE
def get_exact_tau_full(w_hat_n_LF, w_hat_nm1_LF, w_hat_n_HF, w_hat_nm1_HF):

    psi_hat_n_LF = get_psi_hat(w_hat_n_LF)
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF)
    dPsi_n = np.fft.irfft2(psi_hat_n_HF - psi_hat_n_LF)
   
    E_HF_n, Z_HF_n = compute_E_and_Z(w_hat_n_HF, False)   
    E_HF_nm1, Z_HF_nm1 = compute_E_and_Z(w_hat_nm1_HF, False)   
    E_LF_n, Z_LF_n = compute_E_and_Z(w_hat_n_LF, False)   
    E_LF_nm1, Z_LF_nm1 = compute_E_and_Z(w_hat_nm1_LF, False)   

    dE_n = E_HF_n - E_LF_n
    dE_nm1 = E_HF_nm1 - E_LF_nm1
    dZ_n = Z_HF_n - Z_LF_n
    dPsiF = simps(simps(dPsi_n*F, axis), axis)/(2*np.pi)**2

    return -1.0/(2.0*E_LF_n)*((dE_n - dE_nm1)/dt + 2.0*nu*dZ_n + 2.0*mu*dE_n + mu*dPsiF)

#initial model, no dE and dZ contributions
def get_exact_tau(w_hat_n_LF, w_hat_n_HF):

    psi_hat_n_LF = get_psi_hat(w_hat_n_LF)
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF)
    
    w_n_LF = np.fft.irfft2(w_hat_n_LF)
    psi_n_LF = np.fft.irfft2(psi_hat_n_LF)
    dPsi_n = np.fft.irfft2(psi_hat_n_HF - psi_hat_n_LF)

    dPsiF = simps(simps(dPsi_n*F, axis), axis)/(2*np.pi)**2
    E_LF = simps(simps(-0.5*psi_n_LF*w_n_LF, axis), axis)/(2*np.pi)**2

    return -mu*dPsiF/(2.0*E_LF)

#initial model, with dE and dZ contributions
def get_exact_tau2(w_hat_n_LF, w_hat_n_HF):

    psi_hat_n_LF = get_psi_hat(w_hat_n_LF)
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF)
    
    w_n_LF = np.fft.irfft2(w_hat_n_LF)
    psi_n_LF = np.fft.irfft2(psi_hat_n_LF)

    w_n_HF = np.fft.irfft2(w_hat_n_HF)
    psi_n_HF = np.fft.irfft2(psi_hat_n_HF)

    dPsi_n = np.fft.irfft2(psi_hat_n_HF - psi_hat_n_LF)

    E_LF = simps(simps(-0.5*psi_n_LF*w_n_LF, axis), axis)
    E_HF = simps(simps(-0.5*psi_n_HF*w_n_HF, axis), axis)
    Z_LF = simps(simps(0.5*w_n_LF**2, axis), axis)
    Z_HF = simps(simps(0.5*w_n_HF**2, axis), axis)

    dPsiF = simps(simps(dPsi_n*F, axis), axis)
    dE = E_HF - E_LF
    dZ = Z_HF - Z_LF

    return 1.0/(2.0*E_LF)*(-2.0*nu*dZ - 2.0*mu*dE - mu*dPsiF)

#eddy viscosity model, no dE and dZ contributions
def get_exact_tau3(w_hat_n_LF, w_hat_n_HF):

    psi_hat_n_LF = get_psi_hat(w_hat_n_LF)
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF)
    
    w_n_LF = np.fft.irfft2(w_hat_n_LF)
    dPsi_n = np.fft.irfft2(psi_hat_n_HF - psi_hat_n_LF)

    dPsiF = simps(simps(dPsi_n*F, axis), axis)
    Z_LF = simps(simps(0.5*w_n_LF**2, axis), axis)

    return -mu*dPsiF/(2.0*Z_LF)

#eddy viscosity model, with dE and dZ contributions
def get_exact_tau4(w_hat_n_LF, w_hat_n_HF):

    psi_hat_n_LF = get_psi_hat(w_hat_n_LF)
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF)
    
    w_n_LF = np.fft.irfft2(w_hat_n_LF)
    psi_n_LF = np.fft.irfft2(psi_hat_n_LF)

    w_n_HF = np.fft.irfft2(w_hat_n_HF)
    psi_n_HF = np.fft.irfft2(psi_hat_n_HF)

    dPsi_n = np.fft.irfft2(psi_hat_n_HF - psi_hat_n_LF)

    E_LF = simps(simps(-0.5*psi_n_LF*w_n_LF, axis), axis)
    E_HF = simps(simps(-0.5*psi_n_HF*w_n_HF, axis), axis)
    Z_LF = simps(simps(0.5*w_n_LF**2, axis), axis)
    Z_HF = simps(simps(0.5*w_n_HF**2, axis), axis)

    dPsiF = simps(simps(dPsi_n*F, axis), axis)
    dE = E_HF - E_LF
    dZ = Z_HF - Z_LF

    return 1.0/(2.0*Z_LF)*(-2.0*nu*dZ - 2.0*mu*dE - mu*dPsiF)

#Zanna model, no dE and dZ contributions
def get_exact_tau_Zanna(w_hat_n_LF, lhs_hat_n_LF, w_hat_n_HF):

    psi_hat_n_LF = get_psi_hat(w_hat_n_LF)
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF)
    dPsi_n = np.fft.irfft2(psi_hat_n_HF - psi_hat_n_LF)
    
    w_n_LF = np.fft.irfft2(w_hat_n_LF)
    lhs_n_LF = np.fft.irfft2(lhs_hat_n_LF)

    dPsiF = simps(simps(dPsi_n*F, axis), axis)
    denom = simps(simps(w_n_LF*lhs_n_LF, axis), axis)

    return -mu*dPsiF/denom

#compute the energy and enstrophy at t_n
def compute_E_and_Z(w_hat_n, verbose=True):
    
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)
    
    e_n = -0.5*psi_n*w_n
    z_n = 0.5*w_n**2

    E = simps(simps(e_n, axis), axis)/(2*np.pi)**2
    Z = simps(simps(z_n, axis), axis)/(2*np.pi)**2

    if verbose:
        print 'Energy = ', E, ', enstrophy = ', Z
    return E, Z

#compute the (temporal) correlation coeffient 
def corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))

"""
***************************
* M A I N   P R O G R A M *
***************************
"""

import numpy as np
import matplotlib.pyplot as plt
import os, cPickle
import h5py
from drawnow import drawnow
from scipy.integrate import simps
from itertools import combinations, chain
import sys

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

#plt.close('all')
#plt.rcParams['image.cmap'] = 'seismic'

#number of gridpoints in 1D
I = 7
N = 2**I

#2D grid
h = 2*np.pi/N
axis = h*np.arange(1, N+1)
axis = np.linspace(0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, N/2+1]) + 0.0j
ky = np.zeros([N, N/2+1]) + 0.0j

for i in range(N):
    for j in range(N/2+1):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]

k_squared = kx**2 + ky**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = N/3
Ncutoff_LF = 2**(I-1)/3 

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time (in days) + time step
t = 250.0*day
#t_end = t + 8.0*365*day
t_end = 500.0*day
t_data = 500.0*day

dt = 0.01
n_steps = np.ceil((t_end-t)/dt).astype('int')

#############
# USER KEYS #
#############

sim_ID = 'tau_EZ_PE_HF'
#store_frame_rate = np.floor(0.05*day/dt).astype('int')
store_frame_rate = 1
plot_frame_rate = np.floor(0.25*day/dt).astype('int')
corr_frame_rate = np.floor(0.25*day/dt).astype('int')
S = np.floor(n_steps/store_frame_rate).astype('int')

tau_E_max = 1.0
tau_Z_max = 1.0

state_store = False 
restart = True
store = True 
store_fig = False 
plot = True
corr = False
smooth = False
eddy_forcing_type = 'tau_ortho'
binning_type = 'global'

if sim_ID == 'tau_EZ' or sim_ID == 'tau_EZ_PE_HF':
    print 'Using HF nu_LF'
    nu_LF = 1.0/(day*Ncutoff**2*decay_time_nu)
elif sim_ID == 'tau_EZ_nu_LF' or sim_ID == 'tau_EZ_nu_LF_PE_HF':
    print 'Using LF nu_LF'
    nu_LF = 1.0/(day*Ncutoff_LF**2*decay_time_nu)
else:
    print '****************'
    print 'nu_LF not set'
    print '****************'
    import sys; sys.exit()

sim_number = sys.argv[1]
store_ID = sim_ID + '_' + binning_type + '_' + sim_number 

###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#QoI to store, First letter in caps implies an NxN field, otherwise a scalar 

#training data QoI
#QoI = ['e_HF', 'z_HF', 's_HF', 'dE', 'dZ', 'e_LF', 'z_LF', 's_LF', 'e_UP', 'z_UP', 's_UP', 'tau_E', 'tau_Z', 't']
QoI = ['z_np1_HF', 'e_np1_HF', 'z_np1_LF', 'e_np1_LF', \
       'z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF', 'sprime_n_LF', 'zprime_n_LF', \
       'z_n_UP', 'e_n_UP', 'tau_E', 'tau_Z', 't']

#prediction data QoI
#QoI = ['e_HF', 'z_HF', 'e_LF', 'z_LF', 'e_UP', 'z_UP', 'tau_E', 'tau_Z', 'rho', 't']
Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N'] = N
    
    for q in range(Q):
        
        #a field
        if QoI[q][0].isupper():
            samples[QoI[q]] = np.zeros([S, N, N/2+1]) + 0.0j
        #a scalar
        else:
            samples[QoI[q]] = np.zeros(S)

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y);
F_hat = np.fft.rfft2(F);

if restart == True:
    
    state = cPickle.load(open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t/day, 1)) + '.pickle'))
    for key in state.keys():
        print key
        vars()[key] = state[key]

    #initialize the unparameterized solution from the LF model
    #w_hat_n_UP = np.copy(w_hat_n_LF)
    #w_hat_nm1_UP = np.copy(w_hat_nm1_LF)
    #VgradW_hat_nm1_UP = np.copy(VgradW_hat_nm1_LF)
        
else:
    
    #initial condition
    w = np.sin(4.0*x)*np.sin(4.0*y) + 0.4*np.cos(3.0*x)*np.cos(3.0*y) + \
        0.3*np.cos(5.0*x)*np.cos(5.0*y) + 0.02*np.sin(x) + 0.02*np.cos(y)

    #initial Fourier coefficients at time n and n-1
    w_hat_n_HF = P*np.fft.rfft2(w)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    
    w_hat_n_LF = P_LF*np.fft.rfft2(w)
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    
    w_hat_n_UP = P_LF*np.fft.rfft2(w)
    w_hat_nm1_UP = np.copy(w_hat_n_UP)
    
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat(w_hat_n_HF, P)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    VgradW_hat_n_LF = compute_VgradW_hat(w_hat_n_LF, P_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

    VgradW_hat_n_UP = np.copy(VgradW_hat_n_LF)
    VgradW_hat_nm1_UP = np.copy(VgradW_hat_nm1_LF)

#####################
# load binning data #
#####################i

if eddy_forcing_type == 'binned':
    
    fname = HOME + '/samples/' + sim_ID + '_exact_training_t_' + str(np.around(t_data/day, 1)) + '.hdf5'

    print 'Loading', fname

    h5f = h5py.File(fname, 'r')

    print h5f.keys()

    #########################
    N_c = int(sys.argv[2]) 

    S_train = h5f['t'].size
    S_extrapolate = 0 

    idx = 0

    lag = []
    for i in range(N_c):
        lag.append(int(sys.argv[3+i]))
    
    covariates = []
    for i in range(N_c):
        covariates.append(sys.argv[3+N_c+i])

    #spatially constant lag per covariate
    lags = np.zeros(N_c).astype('int')
    for i in range(N_c):
        lags[i] = lag[i]
    
    max_lag = np.max(lags)
    min_lag = np.min(lags)
    j3 = min_lag*store_frame_rate
    
    print '***********************'
    print 'Parameters'
    print '***********************'
    print 'Sim number =', sim_number
    print 'Covariates =', covariates
    print 'Lags =', lag
    print '***********************'

    c_i = np.zeros([S_train - max_lag - S_extrapolate, N_c])
    r = np.zeros([S_train - max_lag - S_extrapolate])

    for s in range(max_lag, S_train - S_extrapolate):
        
        for i in range(N_c):
            
            if covariates[i] == 'auto':
                c_i[idx, i] = h5f['e_np1_HF'][s-lags[i]] - h5f['e_np1_LF'][s-lags[i]]
            elif covariates[i] == 'tau_sprime':
                c_i[idx, i] = h5f['tau_E'][s-lags[i]]*h5f['sprime_n_LF'][s-lags[i]]
            else:
                c_i[idx, i] = h5f[covariates[i]][s-lags[i]]

       #r[idx] = h5f['dE'][s] 
        r[idx] = h5f['e_np1_HF'][s] - h5f['e_np1_LF'][s] 
        
        idx += 1
    
    #########################
    
    N_bins = 100
    
    print 'Creating Binning object...'
    if binning_type == 'global':
        from binning import *
        delta_bin = Binning(c_i, r.flatten(), 1, N_bins, lags = lags, store_frame_rate = store_frame_rate, verbose=True)
        #if N_c == 1:
        #    delta_bin.compute_surrogate_jump_probabilities(plot = True)
        #    delta_bin.compute_jump_probabilities()
        #    delta_bin.plot_jump_pmfs()
    else:
        from local_binning import *
        delta_bin = Local_Binning(c, r.flatten(), N, N_bins, lags = lags, store_frame_rate = store_frame_rate, verbose=True)
    print 'done'

    delta_bin.print_bin_info()

#############################
# SPECIFY CORRELATION PARAM #
#############################
#To compute the correlation between dE & dZ and some specified set of covariates

if corr == True:

    covars = ['s_n_prime', 'u_n_LF',\
              'tau_E*s_n_prime', \
              '2.0*mu*e_n_LF + 2.0*nu*z_n_LF + 2.0*mu*u_n_LF - 2.0*tau_E*s_n_prime']
    correlation = {}

    correlation['dE'] = []
    correlation['dZ'] = []
    correlation['t'] = []

    for i in range(len(covars)):
        correlation[covars[i]] = []

#############################

#smoothing parameters
tau1 = 1.0; tau2 = 1.0; nu1 = 1.0

#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)
norm_factor_smooth = 1.0/(3.0/(2.0*dt) + tau2 - nu1*k_squared)

j = 0; j2 = 0;  j4 = 0; idx = 0;
T  = []; R = []; Tau_E = []; DE = [] 
energy_HF = []; energy_LF = []; energy_UP = []
enstrophy_HF = []; enstrophy_LF = []; enstrophy_UP = []

tau = 1.0 

#time loop
for n in range(n_steps):
    
    #solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)
        
    #exact eddy forcing
    EF_hat_nm1_exact = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF 
    
    #orthogonal patterns
    psi_hat_n_prime = get_psi_hat_prime(w_hat_n_LF)
    w_hat_n_prime = get_w_hat_prime(w_hat_n_LF)

    #exact tau_E and tau_Z
    tau_E, tau_Z, dE, dZ = get_data_driven_tau_src_EZ(w_hat_n_LF, w_hat_n_HF, P_LF, tau_E_max, tau_Z_max)
    
    #E & Z tracking eddy forcing
    EF_hat_n_ortho_exact = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime 

    ##############
    # covariates #
    ##############
    e_n_UP, z_n_UP, _ = get_EZS(w_hat_n_UP)
    e_n_LF, z_n_LF, s_n_LF = get_EZS(w_hat_n_LF)

    psi_n_LF = np.fft.irfft2(get_psi_hat(w_hat_n_LF))
    u_n_LF = 0.5*simps(simps(psi_n_LF*F, axis), axis)/(2.0*np.pi)**2

    #compute S' and Z'
    sprime_n_LF = e_n_LF**2/z_n_LF - s_n_LF
    zprime_n_LF = z_n_LF - e_n_LF**2/s_n_LF

    ##############

    #SURROGATE eddy forcing
    if eddy_forcing_type == 'binned':
    
        if n >= max_lag*store_frame_rate:
            
            if j3 >= min_lag*store_frame_rate:
                j3 = 0

                c_i = delta_bin.get_covar(lags*store_frame_rate)
                r = delta_bin.get_r_ip1(c_i)[0] 
                #r = delta_bin.get_sub_mean_r_ip1(c_i)[0] 
        else:
            r = dE 

        #covar = np.zeros([N**2, N_c])
        covar = np.zeros([1, N_c])
        
        for i in range(N_c):
            if covariates[i] == 'auto':
                covar[:, i] = r.flatten()
            else:
                covar[:, i] = vars()[covariates[i]].flatten()
        
        delta_bin.append_covar(covar)

        #EF_hat = P_LF*np.fft.rfft2(r)
        EF_hat = EF_hat_n_ortho_exact
        j3 += 1

    elif eddy_forcing_type == 'tau':
        EF_hat = -tau_Z*w_hat_n_LF
    elif eddy_forcing_type == 'tau_eddy_visc':
        EF_hat = -tau_E*(kx**2 + ky**2)*w_hat_n_LF
    elif eddy_forcing_type == 'tau_ortho':
        EF_hat = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime
        #EF_hat_nm1 = -tau_Z*w_hat_n_LF - tau_E*w_hat_n_LF
    elif eddy_forcing_type == 'unparam':
        EF_hat = np.zeros([N, N/2+1])
    elif eddy_forcing_type == 'exact':
        EF_hat = EF_hat_nm1_exact
    else:
        print 'No valid eddy_forcing_type selected'
        import sys; sys.exit()
   
    #########################
    if smooth == True:

        if n < max_lag*store_frame_rate:
            EF_hat_n_smooth = EF_hat
            EF_hat_nm1_smooth = EF_hat
            VgradEF_hat_n_smooth = compute_VgradEF_hat(w_hat_n_LF, EF_hat_n_smooth)
            VgradEF_hat_nm1_smooth = VgradEF_hat_n_smooth
        else:
            VgradEF_hat_n_smooth = compute_VgradEF_hat(w_hat_n_LF, EF_hat_n_smooth)

            EF_hat_np1_smooth = norm_factor_smooth*(2.0/dt*EF_hat_n_smooth - 1.0/(2.0*dt)*EF_hat_nm1_smooth \
                                -2.0*VgradEF_hat_n_smooth + VgradEF_hat_nm1_smooth + tau1*EF_hat)

            EF_hat = EF_hat_np1_smooth
            
            #update variables
            EF_hat_nm1_smooth = np.copy(EF_hat_n_smooth)
            EF_hat_n_smooth = np.copy(EF_hat_np1_smooth)
            VgradEF_hat_nm1_smooth = np.copy(VgradEF_hat_n_smooth)

    #########################

    #LF solve
    w_hat_np1_LF, VgradW_hat_n_LF = get_w_hat_np1(w_hat_n_LF, w_hat_nm1_LF, VgradW_hat_nm1_LF, P_LF, norm_factor_LF, EF_hat)
    
    #unparametrized solve
    w_hat_np1_UP, VgradW_hat_n_UP = get_w_hat_np1(w_hat_n_UP, w_hat_nm1_UP, VgradW_hat_nm1_UP, P_LF, norm_factor_LF)

    t += dt
    j += 1
    j2 += 1
    j4 += 1

    if j == plot_frame_rate and plot == True:
        j = 0

        w_np1_HF = np.fft.irfft2(P_LF*w_hat_np1_HF)
        w_np1_LF = np.fft.irfft2(w_hat_np1_LF)

        T.append(t/day)
        if eddy_forcing_type == 'binned':
            R.append(r)
            Tau_E.append(tau_E)
            DE.append(dE)

        EF_nm1_exact = np.fft.irfft2(EF_hat_nm1_exact)
        EF = np.fft.irfft2(EF_hat)
        if smooth == True:
            EF_np1_smooth = np.fft.irfft2(EF_hat_np1_smooth)

        psi_n_LF = np.fft.irfft2(get_psi_hat(w_hat_n_LF))
        dPsi_n = np.fft.irfft2(get_psi_hat(w_hat_n_HF - w_hat_n_LF))
        print spatial_corr_coef(dPsi_n, psi_n_LF)

        print 'tau_E =', tau_E
        print 'tau_Z =', tau_Z
        E_HF, Z_HF = compute_E_and_Z(P_LF*w_hat_np1_HF)
        E_LF, Z_LF = compute_E_and_Z(w_hat_np1_LF)
        E_UP, Z_UP = compute_E_and_Z(w_hat_np1_UP)
    
        energy_HF.append(E_HF); enstrophy_HF.append(Z_HF)
        energy_LF.append(E_LF); enstrophy_LF.append(Z_LF)
        energy_UP.append(E_UP); enstrophy_UP.append(Z_UP)

        drawnow(draw_stats)
        #drawnow(draw_2w)
        
    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0
        
        if np.mod(n, np.round(day/dt)) == 0:
            print 'n = ', n, ' of ', n_steps

        #################
        # training data #
        #################
    
        e_np1_HF, z_np1_HF, _ = get_EZS(P_LF*w_hat_np1_HF)
        e_np1_LF, z_np1_LF, _ = get_EZS(w_hat_np1_LF)

        samples['e_np1_HF'][idx] = e_np1_HF
        samples['z_np1_HF'][idx] = z_np1_HF
        samples['e_np1_LF'][idx] = e_np1_LF
        samples['z_np1_LF'][idx] = z_np1_LF
        samples['e_n_UP'][idx] = e_n_UP
        samples['z_n_UP'][idx] = z_n_UP
        samples['e_n_LF'][idx] = e_n_LF
        samples['z_n_LF'][idx] = z_n_LF
        samples['s_n_LF'][idx] = s_n_LF
        samples['sprime_n_LF'][idx] = sprime_n_LF
        samples['zprime_n_LF'][idx] = zprime_n_LF
        samples['tau_E'][idx] = tau_E
        samples['tau_Z'][idx] = tau_Z
        
        ###################
        # prediction data #
        ###################
        
        #E_HF, Z_HF = compute_E_and_Z(P_LF*w_hat_np1_HF)
        #E_LF, Z_LF = compute_E_and_Z(w_hat_np1_LF)
        #E_UP, Z_UP = compute_E_and_Z(w_hat_np1_UP)
       
        #samples['e_HF'][idx] = E_HF
        #samples['z_HF'][idx] = Z_HF
        #samples['e_LF'][idx] = E_LF
        #samples['z_LF'][idx] = Z_LF
        #samples['e_UP'][idx] = E_UP
        #samples['z_UP'][idx] = Z_UP
        #samples['tau_E'][idx] = tau_E
        #samples['tau_Z'][idx] = tau_Z
        #
        #psi_n_LF = np.fft.irfft2(get_psi_hat(w_hat_n_LF))
        #dPsi_n = np.fft.irfft2(get_psi_hat(w_hat_n_HF - w_hat_n_LF))
        #samples['rho'][idx] = spatial_corr_coef(dPsi_n, psi_n_LF)

        samples['t'][idx] = t
        
        idx += 1  

    if j4 == corr_frame_rate and corr == True:
        j4 = 0
        
        #if np.mod(n, np.round(day/dt)) == 0:
        print 'n = ', n, ' of ', n_steps

        e_n_LF, z_n_LF, s_n_LF = get_EZS(w_hat_n_LF)
        e_np1_LF, z_np1_LF, s_np1_LF = get_EZS(w_hat_np1_LF)
        e_np1_HF, z_np1_HF, s_np1_HF = get_EZS(P_LF*w_hat_np1_HF)
        
        #compute 0.5*(Psi_LF, F) := U_LF
        psi_n_LF = np.fft.irfft2(get_psi_hat(w_hat_n_LF))
        u_n_LF = 0.5*simps(simps(psi_n_LF*F, axis), axis)/(2.0*np.pi)**2

        #compute S'
        s_n_prime = e_n_LF**2/z_n_LF - s_n_LF
      
        #targets
        dE = (e_np1_HF - e_np1_LF)#/e_np1_LF
        dZ = (z_np1_HF - z_np1_LF)#/z_LF

        correlation['dE'].append(dE)
        correlation['dZ'].append(dZ)

        for i in range(len(covars)):
            correlation[covars[i]].append(eval(covars[i]))

        correlation['t'].append(t)

    #update variables
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    w_hat_n_HF = np.copy(w_hat_np1_HF)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)

    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    w_hat_n_LF = np.copy(w_hat_np1_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)
    
    w_hat_nm1_UP = np.copy(w_hat_n_UP)
    w_hat_n_UP = np.copy(w_hat_np1_UP)
    VgradW_hat_nm1_UP = np.copy(VgradW_hat_n_UP)

####################################

#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:
    
    keys = ['t', 'w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF', \
            'w_hat_nm1_LF', 'w_hat_n_LF', 'VgradW_hat_nm1_LF', \
            'w_hat_nm1_UP', 'w_hat_n_UP', 'VgradW_hat_nm1_UP']
    
    state = {}
    
    for key in keys:
        state[key] = vars()[key]
    
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')
    
    cPickle.dump(state, open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.pickle', 'w'))

####################################

#store the samples
if store == True:
    store_samples_hdf5() 

####################################

#store the drawnow figue to file in order to load at a later and and tweak it
if store_fig == True and plot == True:
    ax = plt.gca()

    if os.path.exists(HOME + '/figures') == False:
        os.makedirs(HOME + '/figures')
    
    #generate random filename
    import uuid
    cPickle.dump(ax, open(HOME + '/figures/fig_' + str(uuid.uuid1())[0:8] + '.pickle', 'w'))

####################################

if corr == True:

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(correlation['t'], (correlation['dE'] - np.mean(correlation['dE']))/(np.std(correlation['dE'])), '--k')
    ax.plot(correlation['t'], (correlation['dZ'] - np.mean(correlation['dZ']))/(np.std(correlation['dZ'])), '--b')

    for i in range(len(covars)):
        correlation['dE_rho_' + covars[i]] = corr_coef(correlation['dE'], correlation[covars[i]]) 
        correlation['dZ_rho_' + covars[i]] = corr_coef(correlation['dZ'], correlation[covars[i]]) 

        ax.plot(correlation['t'], (correlation[covars[i]] - np.mean(correlation[covars[i]]))/np.std(correlation[covars[i]]), label=covars[i])

        print 'dE_rho_' + covars[i], correlation['dE_rho_' + covars[i]]
        print 'dZ_rho_' + covars[i], correlation['dZ_rho_' + covars[i]]

    leg = plt.legend(loc=0)
    leg.draggable(True)

####################################

plt.show()
