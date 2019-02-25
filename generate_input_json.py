import numpy as np
import json
import os

HOME = os.path.abspath(os.path.dirname(__file__))

fname = 'test'
target = ['dE']
#covariates = [['z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF'], ['z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF']]
covariates = [['e_n_LF', 'z_n_LF']]
#covariates = [['auto', 'z_n_LF', 'e_n_LF', 'u_n_LF'], ['auto', 'z_n_LF', 'e_n_LF', 'u_n_LF']]
#covariates = [['r_tau_E*sprime_n_LF', 'z_n_LF', 'e_n_LF', 'u_n_LF'], ['r_tau_Z*zprime_n_LF', 'z_n_LF', 'e_n_LF', 'u_n_LF']]
#covariates = [['r_tau_E*sprime_n_LF', 'r_tau_E*sprime_n_LF'], ['r_tau_Z*zprime_n_LF', 'r_tau_Z*zprime_n_LF']]

lag = [[1, 1], [1, 1]]

extrap_ratio = [1.0, 1.0]

N_surr = len(target)

fpath = HOME + '/inputs/' + fname + '.json'

print 'Generating input file', fpath

if os.path.isfile(fpath) == True:
    os.system('rm ' + fpath)

fp = open(fpath, 'a')
fp.write('%d\n' % N_surr)

for i in range(len(target)):
    json_in = {}
    json_in['target'] = target[i]
    json_in['covariates'] = covariates[i]
    json_in['N_c'] = len(covariates[i]) 
    json_in['lag'] = lag[i]
    json_in['extrap_ratio'] = extrap_ratio[i]

    json.dump(json_in, fp)
    fp.write('\n')

    print json_in
