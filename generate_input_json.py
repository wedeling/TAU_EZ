import numpy as np
import json
import os

HOME = os.path.abspath(os.path.dirname(__file__))

#name of the generated input file
input_file = 'T3'

#surrogate targets (Delta E and Delta Z)
target = ['dE', 'dZ']

#specify an empty target if generating training set
#target = []

#choose the conditioning variables (variable names must match those in the main file:tau_ez_ocean.py)
covariates = [['z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF'], ['z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF']]

#per conditioning variable, choose the time lag in number of Delta t
lag = [[1, 1, 1, 1], [1, 1, 1, 1]]

#alpha: the extrapolation ratio (e.g. 0.9 = use 90% of training data)
extrap_ratio = [0.7, 0.7]

#number of surrogates to be constructed
N_surr = len(target)
fpath = HOME + '/inputs/' + input_file + '.json'

print 'Generating input file', fpath

#remove input file if it already exists
if os.path.isfile(fpath) == True:
    os.system('rm ' + fpath)

fp = open(fpath, 'a')
fp.write('%d\n' % N_surr)

#simulation flags
flags = {}
flags['input_file'] = input_file
flags['state_store'] = False                #store the state at the end of the simulation
flags['restart'] = True                     #restart from previously stored state
flags['store'] = True                       #store data
flags['plot'] = False                       #plot results while running (required drawnow package)
flags['compute_ref'] = True                 #compute the reference solution as well, leave at True, will automatically turn off in surrogate mode
flags['eddy_forcing_type'] = 'binned'       #choose 'binned' for surrogate mode, choose 'tau_ortho' for training mode

json.dump(flags, fp)
fp.write('\n')

print flags

#write input file
for i in range(len(target)):
    json_in = {}
    json_in['target'] = target[i]
    json_in['covariates'] = covariates[i]
    json_in['N_c'] = len(covariates[i]) 
    json_in['lag'] = lag[i]
    json_in['extrap_ratio'] = extrap_ratio[i]
    json_in['training_data'] = 'precomputed_training_t_3170.0'

    json.dump(json_in, fp)
    fp.write('\n')

    print json_in
