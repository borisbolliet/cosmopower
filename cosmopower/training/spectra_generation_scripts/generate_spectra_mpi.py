import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
import yaml
import os
from classy_sz import Class
print("%d of %d" % (comm.Get_rank(), comm.Get_size()))

n_samples = 12 #int(sys.argv[1])
n_processes  = 4 #int(sys.argv[2])
n_samples_per_process = int(n_samples/n_processes)
print('doing %d times %d calculations'%(n_processes,n_samples_per_process))

derived_params_names = ['h',
                        'sigma8',
                        'YHe',
                        'z_reio',
                        'Neff',
                        'tau_rec',
                        'z_rec',
                        'rs_rec',
                        'ra_rec',
                        'tau_star',
                        'z_star',
                        'rs_star',
                        'ra_star']


# the running process (index in job array)
# idx_process = sys.argv[1]
idx_process = comm.Get_rank() + 1

path_to_cosmopower_dir = '/Users/boris/Work/CLASS-SZ/SO-SZ/cosmopower'

data_dir_name = 'ACTPol_lite_DR4_baseLCDM_taup_hip'

cobaya_yaml_file = path_to_cosmopower_dir+'/cosmopower/training/spectra_generation_scripts/yaml_files/ACTPol_lite_DR4_baseLCDM_taup_hip.yaml'
with open(cobaya_yaml_file) as f:
    dict_from_yaml_file = yaml.load(f,Loader=yaml.FullLoader)


try:
    os.mkdir(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name)
except FileExistsError:
    print("File exist")

try:
    os.mkdir(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name+'/TT')
except FileExistsError:
    print("File exist")

try:
    os.mkdir(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name+'/TE')
except FileExistsError:
    print("File exist")

try:
    os.mkdir(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name+'/EE')
except FileExistsError:
    print("File exist")

try:
    os.mkdir(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name+'/PP')
except FileExistsError:
    print("File exist")

# get path to the folder of this script
# folder_path = os.path.abspath(os.path.dirname(__file__))
folder_path = path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name


lmax = dict_from_yaml_file['theory']['classy']['extra_args']['l_max_scalars']
# load parameter file for this process index
params_lhs  = np.load(folder_path+'/LHS_parameter_file_{}.npz'.format(idx_process))



def spectra_generation(idx_sample):
    class_params_dict = {}

    for k in params_lhs:
        class_params_dict[k] = params_lhs[k][idx_sample]
    class_params_dict

    cosmo = Class()

    # Define cosmology (what is not specified will be set to CLASS default parameters)
    params = {'output': 'tCl pCl lCl mPk',
              'lensing': 'yes',
              'l_max_scalars': lmax,
              'perturbations_verbose':0
              }

    cosmo.set(params)
    cosmo.set(class_params_dict)
    # cosmo.set(classy_precision)
    cosmo.compute()
    cls = cosmo.lensed_cl(lmax=lmax)

    powers  = cls
    clTT    = powers['tt'][2:]
    clTE    = powers['te'][2:]
    clEE    = powers['ee'][2:]
    clPP    = powers['pp'][2:]

    derp = {}
    for p in derived_params_names:
        derp.update(cosmo.get_current_derived_parameters([p]))

    cosmo_array_tt = np.hstack(([params_lhs[k][idx_sample] for k in params_lhs],[derp[k] for k in derp.keys()], clTT))
    file_tt           = open(folder_path+'/TT/cls_tt_nointerp_{}.dat'.format(idx_process),'ab')
    np.savetxt(file_tt, [cosmo_array_tt])
    file_tt.close()

    cosmo_array_te = np.hstack(([params_lhs[k][idx_sample] for k in params_lhs],[derp[k] for k in derp.keys()], clTE))
    file_te           = open(folder_path+'/TE/cls_te_nointerp_{}.dat'.format(idx_process),'ab')
    np.savetxt(file_te, [cosmo_array_te])
    file_te.close()

    cosmo_array_ee = np.hstack(([params_lhs[k][idx_sample] for k in params_lhs],[derp[k] for k in derp.keys()], clEE))
    file_ee           = open(folder_path+'/EE/cls_ee_nointerp_{}.dat'.format(idx_process),'ab')
    np.savetxt(file_ee, [cosmo_array_ee])
    file_ee.close()

    cosmo_array_pp = np.hstack(([params_lhs[k][idx_sample] for k in params_lhs],[derp[k] for k in derp.keys()], clPP))
    file_pp           = open(folder_path+'/PP/cls_pp_nointerp_{}.dat'.format(idx_process),'ab')
    np.savetxt(file_pp, [cosmo_array_pp])
    file_pp.close()



# loop over parameter sets in parameter file corresponding to the running process
for i in range(n_samples_per_process):
    spectra_generation(i)