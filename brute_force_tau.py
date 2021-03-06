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

def draw_2w():
    plt.subplot(121)
    plt.xlabel(r'$t\;[days]$', fontsize=14)
    #plt.plot(T, DE, label=r'$\Delta E$')
    #plt.plot(T, R_DE, '--' , label=r'$\widetilde{\Delta E}$')
    #plt.plot(T, Tau_E, label=r'$\tau E$')
    plt.plot(T, R_tau_E, '--' , label=r'$\widetilde{\tau E}$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=0, fontsize=14)
    plt.subplot(122)
    plt.xlabel(r'$t\;[days]$', fontsize=14)
    #plt.plot(T, DZ, label=r'$\Delta Z$')
    #plt.plot(T, R_DZ, '--', label=r'$\widetilde{\Delta Z}$')
    #plt.plot(T, Tau_Z, label=r'$\tau Z$')
    plt.plot(T, R_tau_Z, '--', label=r'$\widetilde{\tau Z}$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=0, fontsize=14)
    #plt.hist([DZ, R], 20, label=[r'$\Delta Z$', r'$\widetilde{\Delta_Z}$'])
    #plt.legend(loc='upper right')
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
    #plt.tight_layout()
    
def movie(s):
    plt.clf()
    plt.subplot(121, xlabel=r't [days]', title=r'energy E')
    plt.plot(samples['t'][0:s]/day, samples['e_n_LF'][0:s], 'ro', alpha=0.4, label=r'$\mathrm{reduced}$')
    plt.plot(samples['t'][0:s]/day, samples['e_n_HF'][0:s], 'b', linewidth=2, label=r'$\mathrm{reference}$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=2)

#    #plt.plot(samples['t'][0:s]/day, samples['z_n_UP'][0:s], label=r'$Z^{UP}$')
#    plt.subplot(132, xlabel=r't [days]', title=r'enstrophy Z')
#    plt.plot(samples['t'][0:s]/day, samples['z_n_LF'][0:s], 'ro', alpha=0.4, label=r'$\mathrm{reduced}$')
#    plt.plot(samples['t'][0:s]/day, samples['z_n_HF'][0:s], 'b', linewidth=2, label=r'$\mathrm{reference}$')
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    plt.legend(loc=2)
    #
    plt.subplot(122, xlabel='x', ylabel='y', title=r'$\tau_E \omega$')
    EF = np.fft.irfft2(samples['EF_hat'][s,:,:])
    EF_a = np.min(EF); EF_b = np.max(EF)
    plt.contourf(x, y, 2*(EF - EF_a)/(EF_b - EF_a) - 1.0, np.linspace(-1, 1, 100))

#    plt.subplot(133, xlabel='x', ylabel='y', title=r'reference eddy forcing')
#    EF_exact = np.fft.irfft2(samples['EF_hat_exact'][s,:,:])
#    EF_a = np.min(EF_exact); EF_b = np.max(EF_exact)
#    plt.contourf(x, y, 2*(EF_exact - EF_a)/(EF_b - EF_a) - 1.0, np.linspace(-1, 1, 100))

    plt.tight_layout()    

#return the fourier coefs of the stream function
def get_psi_hat(w_hat_n):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0

    return psi_hat_n

###############################
# DATA-DRIVEN TAU SUBROUTINES #
###############################
    
def get_data_driven_tau_src_E(w_hat_n_LF, w_hat_n_HF, P, tau_max_E):
    
    E_LF, Z_LF, S_LF = get_EZS(w_hat_n_LF)

    E_HF = compute_E(P*w_hat_n_HF)

    dE = (E_HF - E_LF)#/E_LF

    tau_E = tau_max_E*np.tanh(dE/E_LF)
    
    return tau_E, dE

def get_data_driven_tau_src_EZ(w_hat_n_LF, w_hat_n_HF, P, tau_max_E, tau_max_Z):
    
    E_LF, Z_LF, S_LF = get_EZS(w_hat_n_LF)

    src_E = E_LF**2/Z_LF - S_LF
    src_Z = -E_LF**2/S_LF + Z_LF

    E_HF = compute_E(P*w_hat_n_HF)
    Z_HF = compute_Z(P*w_hat_n_HF)

    dE = (E_HF - E_LF)#/E_LF
    dZ = (Z_HF - Z_LF)#/Z_LF

    tau_E = tau_max_E*np.tanh(dE/E_LF)*np.sign(src_E)
    tau_Z = tau_max_Z*np.tanh(dZ/Z_LF)*np.sign(src_Z)
    
    return tau_E, tau_Z, dE, dZ

def get_surrogate_tau_src_EZ(w_hat_n_LF, r, tau_max_E, tau_max_Z):
    
    E_LF, Z_LF, S_LF = get_EZS(w_hat_n_LF)

    src_E = E_LF**2/Z_LF - S_LF
    src_Z = -E_LF**2/S_LF + Z_LF

    dE = r['dE'] 
    dZ = r['dZ']

    tau_E = tau_max_E*np.tanh(dE/E_LF)*np.sign(src_E)
    tau_Z = tau_max_Z*np.tanh(dZ/Z_LF)*np.sign(src_Z)
    
    return tau_E, tau_Z

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
import json
import matplotlib.animation as animation

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

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

#start, end time, end time of data (training period), time step
t = 250.0*day
t_end = t + 8.0*365*day
t_end = 350.0*day
t_data = t + 8.0*365.0*day 
dt = 0.01
n_steps = np.ceil((t_end-t)/dt).astype('int')

#############
# USER KEYS #
#############

#simulation name
sim_ID = 'tau_EZ_PE_HF'
#framerate of storing data, plotting results, computing correlations
#store_frame_rate = 1
store_frame_rate = np.floor(0.25*day/dt).astype('int')
plot_frame_rate = np.floor(0.25*day/dt).astype('int')
corr_frame_rate = np.floor(0.25*day/dt).astype('int')
#length of data array
S = np.floor(n_steps/store_frame_rate).astype('int')

#user-specified parameter of tau_E and tau_Z terms
tau_E_max = 1.0
tau_Z_max = 1.0

#flags 
state_store = False     #store the state at the end
restart = True          #restart from prev state
store = True            #store data
store_fig = False       #store figure object
plot = False             #plot results while running, requires drawnow package
corr = False            #compute and store correlations
make_movie = True

eddy_forcing_type = 'tau_ortho'    #which eddy forcing to use

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
store_ID = sim_ID + '_' + sim_number 

###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#QoI to store, First letter in caps implies an NxN field, otherwise a scalar 
QoI = ['z_n_HF', 'e_n_HF', 'z_n_UP', 'e_n_UP', \
       'z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF', 'v_n_LF', 'o_n_LF', \
       'sprime_n_LF', 'zprime_n_LF', \
       'tau_E', 'tau_Z', 'r_tau_E', 'r_tau_Z', 't', \
       'EF_hat', 'EF_hat_exact']
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

    ###########################
    # load the reference data #
    ###########################
    fname = HOME + '/samples/' + sim_ID + '_exact_training_t_' + str(np.around(t_data/day, 1)) + '.hdf5'

    print 'Loading', fname

    h5f = h5py.File(fname, 'r')

    print h5f.keys()

    #####################################
    # read the inputs for the surrogate #
    #####################################
    fpath = sys.argv[2]
    fp = open(fpath, 'r')
    N_surr = int(fp.readline())
    inputs = []
    
    for i in range(N_surr):
        inputs.append(json.loads(fp.readline()))
        
    print '****************************'
    print 'Creating', N_surr, ' surrogates'
    print '****************************'

    #########################
    # create the surrogates #
    #########################
    
    surrogate = {}; N_c = {}; covariates = {}; lags = {}; j3 = {}
    for j in range(N_surr):

        param = inputs[j]
        
        #read the dict
        target = param['target']
        N_c[target] = param['N_c'] 
        covariates[target] = param['covariates']
        lag = param['lag']
        extrap_ratio = param['extrap_ratio']

        #reduce training data if extrap_ratio < 1 (to test extrapolative capability of surrogate)
        S_tot = h5f['t'].size
        S_train = np.int(extrap_ratio*h5f['t'].size)
        S_extrap = S_tot - S_train

        #spatially constant lag per covariate
        lags[target] = np.zeros(N_c[target]).astype('int')
        for i in range(N_c[target]):
            lags[target][i] = lag[i]
        
        max_lag = np.max(lags[target])
        min_lag = np.min(lags[target])
        j3[target] = 0#min_lag*store_frame_rate
        
        print '***********************'
        print 'Parameters'
        print '***********************'
        print 'Sim number =', sim_number
        print 'Target =', target
        print 'Covariates =', covariates[target]
        print 'Excluding', S_extrap, ' points from training set' 
        print 'Lags =', lags[target]
        print '***********************'

        #covariates
        c_i = np.zeros([S_train - max_lag, N_c[target]])
        i1 = max_lag; i2 = S_tot - S_extrap

        for i in range(N_c[target]):

            lag_i = lags[target][i] 

            if covariates[target][i] == 'auto' and target == 'dE':
                c_i[:, i] = h5f['e_n_HF'][i1-lag_i:i2-lag_i] - h5f['e_n_LF'][i1-lag_i:i2-lag_i]
            elif covariates[target][i] == 'auto' and target == 'dZ':
                c_i[:, i] = h5f['z_n_HF'][i1-lag_i:i2-lag_i] - h5f['z_n_LF'][i1-lag_i:i2-lag_i]
            elif covariates[target][i] == 'r_tau_E*sprime_n_LF':
                c_i[:, i] = h5f['tau_E'][i1-lag_i:i2-lag_i]*h5f['sprime_n_LF'][i1-lag_i:i2-lag_i]
            elif covariates[target][i] == 'r_tau_Z*zprime_n_LF':
                c_i[:, i] = h5f['tau_Z'][i1-lag_i:i2-lag_i]*h5f['zprime_n_LF'][i1-lag_i:i2-lag_i]
            else:
                c_i[:, i] = h5f[covariates[target][i]][i1-lag_i:i2-lag_i]

        if target == 'dZ':
            r = h5f['z_n_HF'][i1:i2] - h5f['z_n_LF'][i1:i2]
        elif target == 'dE':
            r = h5f['e_n_HF'][i1:i2] - h5f['e_n_LF'][i1:i2]
        
        #########################
        
        N_bins = 10

        print 'Creating Binning object...'
        from binning import *
        surrogate[target] = Binning(c_i, r.flatten(), 1, N_bins, lags = lags[target], store_frame_rate = store_frame_rate, verbose=True)
        #surrogate[target].plot_samples_per_bin()
        #if N_c == 1:
        #    surrogate[target].compute_surrogate_jump_probabilities(plot = True)
        #    surrogate[target].compute_jump_probabilities()
        #    surrogate[target].plot_jump_pmfs()
        print 'done'

        surrogate[target].print_bin_info()

#############################
# SPECIFY CORRELATION PARAM #
#############################
#To compute the correlation between dE & dZ and some specified set of covariates

if corr == True:

    covars = ['z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF', 'v_n_LF', 'o_n_LF', 'sprime_n_LF', 'zprime_n_LF', 'tau_E*sprime_n_LF', 'tau_Z*zprime_n_LF']
    correlation = {}

    correlation['dE'] = []
    correlation['dZ'] = []
    correlation['t'] = []

    for i in range(len(covars)):
        correlation[covars[i]] = []

#############################

#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)

j = 0; j2 = 0;  j4 = 0; idx = 0;
T  = []; R_DE = []; R_DZ = []; Tau_E = []; DE = []; Tau_Z = []; DZ = []; R_tau_E = []; R_tau_Z = []  
energy_HF = []; energy_LF = []; energy_UP = []
enstrophy_HF = []; enstrophy_LF = []; enstrophy_UP = []

if make_movie == True:
    fig = plt.figure(figsize=[6.7, 4])

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
    tau_E, dE = get_data_driven_tau_src_E(w_hat_n_LF, w_hat_n_HF, P_LF, tau_E_max)
    
    #E & Z tracking eddy forcing
    #EF_hat_n_ortho = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime 

    ##############
    # covariates #
    ##############
    e_n_UP, z_n_UP, _ = get_EZS(w_hat_n_UP)
    e_n_LF, z_n_LF, s_n_LF = get_EZS(w_hat_n_LF)

    psi_n_LF = np.fft.irfft2(get_psi_hat(w_hat_n_LF))
    u_n_LF = 0.5*simps(simps(psi_n_LF*F, axis), axis)/(2.0*np.pi)**2
    w_n_LF = np.fft.irfft2(w_hat_n_LF)
    v_n_LF = 0.5*simps(simps(w_n_LF*F, axis), axis)/(2.0*np.pi)**2
    nabla2_w_n_LF = np.fft.irfft2(k_squared*w_hat_n_LF)
    o_n_LF = 0.5*simps(simps(nabla2_w_n_LF*w_n_LF, axis), axis)/(2.0*np.pi)**2

    #compute S' and Z'
    sprime_n_LF = e_n_LF**2/z_n_LF - s_n_LF
    zprime_n_LF = z_n_LF - e_n_LF**2/s_n_LF

    ##############

    #SURROGATE eddy forcing
    if eddy_forcing_type == 'binned':
   
        r = {}

        for target in surrogate.keys():

            if n >= np.max(lags[target])*store_frame_rate:
                
                if j3[target] >= np.min(lags[target])*store_frame_rate:
                    j3[target] = 0

                    c_i = surrogate[target].get_covar(lags[target]*store_frame_rate)
                    r[target] = surrogate[target].get_r_ip1(c_i)[0]
            else:
                r[target] = eval(target) 
                r_tau_E = tau_E
                r_tau_Z = tau_Z

            #covar = np.zeros([N**2, N_c])
            covar = np.zeros([1, N_c[target]])
            
            for i in range(N_c[target]):
                if covariates[target][i] == 'auto':
                    covar[:, i] = r[target]#.flatten()
                else:
                    #covar[:, i] = vars()[covariates[i]].flatten()
                    covar[:, i] = eval(covariates[target][i])
            
            surrogate[target].append_covar(covar)

            j3[target] += 1

        r_tau_E, r_tau_Z = get_surrogate_tau_src_EZ(w_hat_n_LF, r, tau_E_max, tau_Z_max)

        #use the surrogate orthogonal-pattern eddy forcing
        EF_hat =  -r_tau_E*psi_hat_n_prime - r_tau_Z*w_hat_n_prime 

    elif eddy_forcing_type == 'tau_ortho':
        #EF_hat = -tau_E*psi_hat_n_prime - tau_Z*w_hat_n_prime
        EF_hat = -tau_E*w_hat_n_LF
    elif eddy_forcing_type == 'unparam':
        EF_hat = np.zeros([N, N/2+1])
    elif eddy_forcing_type == 'exact':
        EF_hat = EF_hat_nm1_exact
    else:
        print 'No valid eddy_forcing_type selected'
        import sys; sys.exit()
   
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
            R_DE.append(r['dE'])
            R_DZ.append(r['dZ'])
            Tau_E.append(tau_E)
            DE.append(dE)
            Tau_Z.append(tau_Z)
            DZ.append(dZ)
            R_tau_E.append(r_tau_E)
            R_tau_Z.append(r_tau_Z)

        EF_nm1_exact = np.fft.irfft2(EF_hat_nm1_exact)
        EF = np.fft.irfft2(EF_hat)

        psi_n_LF = np.fft.irfft2(get_psi_hat(w_hat_n_LF))
        dPsi_n = np.fft.irfft2(get_psi_hat(w_hat_n_HF - w_hat_n_LF))
        print corr_coef(dPsi_n, psi_n_LF)

        print 'tau_E =', tau_E
        #print 'tau_Z =', tau_Z
        E_HF, Z_HF, _ = get_EZS(P_LF*w_hat_np1_HF)
        E_LF, Z_LF, _ = get_EZS(w_hat_np1_LF)
        E_UP, Z_UP, _ = get_EZS(w_hat_np1_UP)
    
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
    
        e_n_HF, z_n_HF, _ = get_EZS(P_LF*w_hat_n_HF)
        e_n_LF, z_n_LF, _ = get_EZS(w_hat_n_LF)

        samples['e_n_HF'][idx] = e_n_HF     
        samples['z_n_HF'][idx] = z_n_HF
        samples['e_n_UP'][idx] = e_n_UP
        samples['z_n_UP'][idx] = z_n_UP
        samples['z_n_LF'][idx] = z_n_LF
        samples['e_n_LF'][idx] = e_n_LF
        samples['u_n_LF'][idx] = u_n_LF
        samples['s_n_LF'][idx] = s_n_LF
        samples['v_n_LF'][idx] = v_n_LF
        samples['o_n_LF'][idx] = o_n_LF
        samples['sprime_n_LF'][idx] = sprime_n_LF
        samples['zprime_n_LF'][idx] = zprime_n_LF
        samples['tau_E'][idx] = tau_E
        samples['tau_Z'][idx] = tau_Z
        #samples['r_tau_E'][idx] = r_tau_E
        #samples['r_tau_Z'][idx] = r_tau_Z
        samples['t'][idx] = t
        
        samples['EF_hat'][idx,:,:] = EF_hat
        samples['EF_hat_exact'][idx,:,:] = EF_hat_nm1_exact
        
        idx += 1  

    if j4 == corr_frame_rate and corr == True:
        j4 = 0
        
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

    print '***************************'

    print '\\begin{table}'
    print '\\centering'
    print '\\begin{tabular}{ccc}'
    print '\\hline\\hline'
    print  '$\mathrm{cond.\;var.}$ & $\Delta E$ & $\Delta Z$ \\\\'
    print '\\hline'

    for i in range(len(covars)):
        correlation['dE_rho_' + covars[i]] = corr_coef(correlation['dE'], correlation[covars[i]]) 
        correlation['dZ_rho_' + covars[i]] = corr_coef(correlation['dZ'], correlation[covars[i]]) 

        ax.plot(correlation['t'], (correlation[covars[i]] - np.mean(correlation[covars[i]]))/np.std(correlation[covars[i]]), label=covars[i])

        print covars[i] + '&' + str(np.around(correlation['dE_rho_' + covars[i]], 4)) + '&' + str(np.around(correlation['dZ_rho_' + covars[i]], 4)) + '\\\\'

    print '\\hline\\hline'
    print '\\end{tabular}'
    print '\\caption{Correlation coefficients. \\label{tab:rho}}'
    print '\\end{table}'
    print '***************************'
    
    leg = plt.legend(loc=0)
    leg.draggable(True)

####################################
    
if make_movie == True:
    ani = animation.FuncAnimation(fig, movie, frames=np.arange(S))
    writer = animation.writers['ffmpeg'](fps=8)
    ani.save('./demo.mp4',writer=writer,dpi=100)

plt.show()
