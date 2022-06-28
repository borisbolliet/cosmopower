# this should be run with:
# from inside: /spectra_generation_scripts/
#$ mpirun -np 4 python 2_generate_spectra_mpi.py -dir ../training_data/ACTPol_lite_DR4_baseLCDM_taup_hip_4_by_20
import argparse
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
import yaml
import os
import glob
from classy_sz import Class
from pkg_resources import resource_filename
import time

# print("%d of %d" % (comm.Get_rank(), comm.Get_size()))

from itertools import groupby

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

with_classy_precision = False


def run(args):
    start = time.time()
    def check_params_and_files():
        print('checking params and files')
    # n_samples = args.n_samples #int(sys.argv[1])
    # n_processes  = args.n_processes #int(sys.argv[2]) # should be the same as number of "-np" processes

    n_processes = comm.Get_size()
    # n_samples_per_process = int(n_samples/n_processes)

    dir = args.dir
    dir = dir.split('/')[len(dir.split('/'))-1]
    n_samples_per_process = dir.split('_')[len(dir.split('_'))-1]
    n_samples_per_process = int(n_samples_per_process)

    boltzmann_verbose = args.boltzmann_verbose
    if boltzmann_verbose is None:
        boltzmann_verbose = 0


    n_process_check = dir.split('_')[len(dir.split('_'))-3]
    n_process_check = int(n_process_check)
    # check we ask for the correct number of samples and processes
    if n_processes != n_process_check:
        print('Wrong number of processes requested.')
        print('This number should be equal to the number of LHs in the directory.')
        print('The correct number is %d.'%n_process_check)
        exit(0)

    print('doing %d (%d) times %d calculations'%(n_processes,n_process_check,n_samples_per_process))


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

    # path_to_cosmopower_dir = '/Users/boris/Work/CLASS-SZ/SO-SZ/cosmopower'

    path_to_cosmopower_dir = resource_filename("cosmopower","/../")


    cobaya_yaml_file = glob.glob(args.dir+"/*.yaml")[0]
    # path_to_cosmopower_dir+'/cosmopower/training/spectra_generation_scripts/yaml_files/ACTPol_lite_DR4_baseLCDM_taup_hip.yaml'
    with open(cobaya_yaml_file) as f:
        dict_from_yaml_file = yaml.load(f,Loader=yaml.FullLoader)


    classy_precision = dict_from_yaml_file['theory']['classy']['extra_args']
    # try:
    #     os.mkdir(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name)
    # except FileExistsError:
    #     print("File exist")

    try:
        os.mkdir(args.dir+'/TT')
    except FileExistsError:
        print("File exist")

    try:
        os.mkdir(args.dir+'/TE')
    except FileExistsError:
        print("File exist")

    try:
        os.mkdir(args.dir+'/EE')
    except FileExistsError:
        print("File exist")

    try:
        os.mkdir(args.dir+'/PP')
    except FileExistsError:
        print("File exist")


    # scratch_path = '/global/cscratch1/sd/bb3028'
    #
    # try:
    #     os.mkdir(scratch_path+data_dir_name+'/TT')
    # except FileExistsError:
    #     print("File exist")
    #
    # try:
    #     os.mkdir(scratch_path+data_dir_name+'/TE')
    # except FileExistsError:
    #     print("File exist")
    #
    # try:
    #     os.mkdir(scratch_path+data_dir_name+'/EE')
    # except FileExistsError:
    #     print("File exist")
    #
    # try:
    #     os.mkdir(scratch_path+data_dir_name+'/PP')
    # except FileExistsError:
    #     print("File exist")


    # get path to the folder of this script
    # folder_path = os.path.abspath(os.path.dirname(__file__))
    folder_path = args.dir


    lmax = dict_from_yaml_file['theory']['classy']['extra_args']['l_max_scalars']
    # load parameter file for this process index
    params_lhs  = np.load(folder_path+'/LHS_parameter_file_{}.npz'.format(idx_process))


    # for k in params_lhs:
    #     n_samples_per_process_from_lhs = len(params_lhs[k])
    #     break
    # if n_samples_per_process_from_lhs != n_samples_per_process:
    #     print('Wrong number of samples requested.')
    #     print('The correct number of samples is %d.'%(n_samples_per_process_from_lhs*n_processes))
    #     exit(0)


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
                  'perturbations_verbose':boltzmann_verbose
                  }

        cosmo.set(params)
        cosmo.set(class_params_dict)
        if with_classy_precision:
            cosmo.set(classy_precision)
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


    n_samples_per_process_start = 0


    if args.restart=='yes':
        nparams = np.shape(params_lhs)[0]
        n_samples_per_process_start = []
        dns = ['TT','TE','EE','PP']
        clns = ['tt','te','ee','pp']
        for (dn,cln) in zip(dns,clns):
            f = np.loadtxt(folder_path+'/%s/cls_%s_nointerp_%s.dat'%(dn,cln,str(idx_process)))
            t_params_f = f[np.shape(f)[0]-1][:nparams]
            for idx_samplep in range(n_samples_per_process):
                t_params_lhs = [params_lhs[k][idx_samplep] for k in params_lhs]
                a_diff = np.asarray(t_params_lhs) - np.asarray(t_params_f)
                if a_diff.all() == 0:
                    print('restarting at sample %d'%(idx_samplep+2))
                    n_samples_per_process_start.append(idx_samplep + 1)
                    break
        # cut the file
        if not all_equal(n_samples_per_process_start):
            print(n_samples_per_process_start)
            print(min(n_samples_per_process_start))
            idn_min = n_samples_per_process_start.index(min(n_samples_per_process_start))
            print(idn_min)
            print(dns[idn_min],clns[idn_min])
            print(dns)
            with open(folder_path+'/%s/cls_%s_nointerp_%s.dat'%(dns[idn_min],clns[idn_min],str(idx_process)), 'r') as fp:
                lines = fp.readlines()
            nlines_min = len(lines)
            print('nlines_min:',nlines_min)

            dns.pop(idn_min)
            clns.pop(idn_min)
            print('new list:',dns,clns)
            for (dn,cln) in zip(dns,clns):
                lines = []
                with open(folder_path+'/%s/cls_%s_nointerp_%s.dat'%(dn,cln,str(idx_process)), 'r') as fp:
                    lines = fp.readlines()
                nlines = len(lines)
                with open(folder_path+'/%s/cls_%s_nointerp_%s.dat'%(dn,cln,str(idx_process)), 'w') as fp:
                    for number, line in enumerate(lines):
                        # # delete line 5 and 8. or pass any Nth line you want to remove
                        # # note list index starts from 0
                        # if number in np.arange(nlines_min,nlines):
                        #     print(number)
                        if number not in np.arange(nlines_min,nlines):
                            fp.write(line)



        # f = np.loadtxt(folder_path+'/%s/cls_%s_nointerp_%s.dat'%('PP','pp',str(idx_process)))

        # # list to store file lines
        # lines = []
        # # read file
        # with open(folder_path+'/%s/cls_%s_nointerp_%s.dat'%('PP','pp',str(idx_process)), 'r') as fp:
        #     # read an store all lines into list
        #     lines = fp.readlines()
        #
        # nlines = len(lines)
        # print(nlines)
        # # Write file
        # with open(folder_path+'/%s/cls_%s_nointerp_%s.dat'%('PP','pp',str(idx_process)), 'w') as fp:
        #     # iterate each line
        #     for number, line in enumerate(lines):
        #         # delete line 5 and 8. or pass any Nth line you want to remove
        #         # note list index starts from 0
        #         if number not in [nlines-1]:
        #             fp.write(line)
        # exit(0)
        # # Write file
        # with open(folder_path+'/%s/cls_%s_nointerp_%s.dat'%('PP','pp',str(idx_process)), 'w') as fp:
        #     # iterate each line
        #     for number, line in enumerate(lines):
        #         # delete line 5 and 8. or pass any Nth line you want to remove
        #         # note list index starts from 0
        #         if number not in [4, 7]:
        #             fp.write(line)


        n_samples_per_process_start = min(n_samples_per_process_start)

        # exit(0)
    else:
        print('Starting computation from first sample.')
        print('If you want to restart from last sample, set -restart "yes".')
        # exit(0)

    if n_samples_per_process_start == n_samples_per_process:
        print('computation finished for process %d'%(comm.Get_rank()+1))
        check_params_and_files()
    else:
        # loop over parameter sets in parameter file corresponding to the running process
        for i in range(n_samples_per_process_start,n_samples_per_process):
            print('doing sample %d/%d in process %d...'%(i+1,n_samples_per_process,comm.Get_rank()+1))
            spectra_generation(i)
            print('sample %d/%d in process %d done.'%(i+1,n_samples_per_process,comm.Get_rank()+1))

        print('computation finished for process %d'%(comm.Get_rank()+1))
        check_params_and_files()
    end = time.time()
    seconds = end - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print ('Time elapsed: {:d}:{:02d}:{:02d}'.format(h, m, s))




def main():
    parser=argparse.ArgumentParser(description="generate spectra")
    parser.add_argument("-dir",help="dir" ,dest="dir", type=str, required=True)
    parser.add_argument("-restart",help="restart" ,dest="restart", type=str, required=False)
    parser.add_argument("-boltzmann_verbose",help="boltzmann_verbose" ,dest="boltzmann_verbose", type=int, required=False)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
	main()
