# this should be run with:
#$ python 1_generate_lhc.py -n_samples_per_process 20 -n_processes 4 -yaml_file yaml_files/ACTPol_lite_DR4_baseLCDM_taup_hip.yaml
import argparse
import numpy as np
import yaml
import os
from classy_sz import Class
from pkg_resources import resource_filename
import pyDOE as pyDOE

set_width_from_plc18 = True
test_lh_boundaries = False
with_classy_precision = False

def run(args):
    n_samples_per_process = args.n_samples_per_process #int(sys.argv[1])
    n_processes  = args.n_processes #int(sys.argv[2]) # should be the same as number of "-np" processes
    n_samples = n_samples_per_process*n_processes

    path_to_training_data_dir = args.path_to_training_data_dir
    print(path_to_training_data_dir)



    path_to_cosmopower_dir = resource_filename("cosmopower","/../")


    cobaya_yaml_file = args.yaml_file
    # path_to_cosmopower_dir+'/cosmopower/training/spectra_generation_scripts/yaml_files/ACTPol_lite_DR4_baseLCDM_taup_hip.yaml'
    # data_dir_name = 'ACTPol_lite_DR4_baseLCDM_taup_hip'

    # print(cobaya_yaml_file.split('/'))
    sp_str = cobaya_yaml_file.split('/')

    data_dir_name = sp_str[len(sp_str)-1]
    yaml_file_name = data_dir_name
    data_dir_name = data_dir_name.replace('.yaml','')



    data_dir_name += '_'+str(n_processes)+'_by_'+str(n_samples_per_process)

    with open(cobaya_yaml_file) as f:
        dict_from_yaml_file = yaml.load(f,Loader=yaml.FullLoader)

    # get path to the folder of this script
    # folder_path = os.path.abspath(os.path.dirname(__file__))
    folder_path = path_to_cosmopower_dir+'/cosmopower/training/training_data'
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print("File exist")

    folder_path = path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print("File exist")

    with open(folder_path+'/'+yaml_file_name, 'w') as file:
        yaml.dump(dict_from_yaml_file, file)







    # it doesnt matter if these params are in the yaml or not.
    # but order needs to be preserved
    cosmo_params = ['logA','n_s','theta_s_1e2','omega_b','omega_cdm','tau_reio']
    cosmo_params_names_in_class = ['ln10^{10}A_s','n_s','100*theta_s','omega_b','omega_cdm','tau_reio']

    # last column of Table 1 of https://arxiv.org/pdf/1807.06209.pdf
    p18_sigmas = [0.014,0.0042,0.00031,0.00015,0.0012,0.0074]
    p18_values = [3.043,0.9652,1.04089,0.02233,0.1198,0.0540]



    param_list_all = []
    param_list_varied = {}
    param_list_varied_cosmo = {}
    param_list_other = []
    pdict = dict_from_yaml_file['params']
    for (k,v) in zip(pdict.keys(),pdict.values()):
        #print(k,v)
        #print(k)
        if 'prior' in v:
            #print('varied param')
    #         print(v['prior'])
            param_list_varied[k] = {}
            param_list_varied[k]['bounds'] ={}
            if ('min' and 'max') in v['prior']:
    #             print(v['prior']['min'])
                param_list_varied[k]['bounds']={'min':v['prior']['min'],'max':v['prior']['max']}
            elif 'dist' in v['prior']:
    #             print(k,v['prior'])
                param_list_varied[k]['bounds']={'min':v['prior']['loc']-5.*v['prior']['scale'],'max':v['prior']['loc']+5.*v['prior']['scale']}
                if k == 'tau_reio':
                    if param_list_varied[k]['bounds']['min']<0:
                        param_list_varied[k]['bounds'] = {'min':0.001,'max':v['prior']['loc']+5.*v['prior']['scale']}

        else:
            param_list_other.append(k)

    for (p,v) in zip(param_list_varied.keys(),param_list_varied.values()):
        if p in cosmo_params:
    #         param_list_varied_cosmo[p]['bounds']={}
    #         param_list_varied_cosmo[p]['bounds']=v
            param_list_varied_cosmo[p]=v
            param_list_varied_cosmo[p]['class_name'] = {}
            param_list_varied_cosmo[p]['class_name'] = cosmo_params_names_in_class[cosmo_params.index(p)]


    if set_width_from_plc18:
        for p in param_list_varied_cosmo.keys():
            param_list_varied_cosmo[p]['bounds']['min'] = p18_values[cosmo_params.index(p)]-20.*p18_sigmas[cosmo_params.index(p)]
            param_list_varied_cosmo[p]['bounds']['max'] = p18_values[cosmo_params.index(p)]+20.*p18_sigmas[cosmo_params.index(p)]
            if p == 'tau_reio':
                if param_list_varied_cosmo[p]['bounds']['min']<0:
                    param_list_varied_cosmo[p]['bounds']['min'] = 0.01

    n_params = len(param_list_varied_cosmo)

    if test_lh_boundaries:
        # Function to generate all binary strings
        def generateAllBinaryStrings(n, arr, i,combs):
            if i == n:
                #print(len(arr),arr)
                combs.append(arr.copy())
                return

            arr[i] = 0
            generateAllBinaryStrings(n, arr, i + 1,combs)

            arr[i] = 1
            generateAllBinaryStrings(n, arr, i + 1,combs)

        n = len(param_list_varied_cosmo.keys())
        arr = [None] * n
        combs = []
        generateAllBinaryStrings(n, arr, 0,combs)

        for ic in range(len(combs)):
            class_params_dict = {}
            for ik,k in enumerate(param_list_varied_cosmo.keys()):
                if combs[ic][ik] == 0:
                    class_params_dict[param_list_varied_cosmo[k]['class_name']] = param_list_varied_cosmo[k]['bounds']['min']
                else:
                    class_params_dict[param_list_varied_cosmo[k]['class_name']] = param_list_varied_cosmo[k]['bounds']['max']
            print(class_params_dict)

            cosmo = Class()
            lmax = dict_from_yaml_file['theory']['classy']['extra_args']['l_max_scalars']
            # Define cosmology (what is not specified will be set to CLASS default parameters)
            params = {'output': 'tCl pCl lCl',
                      'lensing': 'yes',
                      'l_max_scalars': lmax,
                      'perturbations_verbose':3
                      }

            cosmo.set(params)
            cosmo.set(class_params_dict)
            if with_classy_precision:
                cosmo.set(classy_precision)
            cosmo.compute()
            print('done')



    print('doing %d times %d calculations'%(n_processes,n_samples_per_process))

    for (k,v) in zip(param_list_varied_cosmo.keys(),param_list_varied_cosmo.values()):
        param_list_varied_cosmo[k]['array'] = {}
        param_list_varied_cosmo[k]['array'] = np.linspace(param_list_varied_cosmo[k]['bounds']['min'],param_list_varied_cosmo[k]['bounds']['max'],n_samples)

    all_params_list = []
    for (k,v) in zip(param_list_varied_cosmo.keys(),param_list_varied_cosmo.values()):
        all_params_list.append(v['array'])

    AllParams = np.vstack(all_params_list)
    lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
    idx = (lhd * n_samples).astype(int)

    AllCombinations = np.zeros((n_samples, n_params))
    for i in range(n_params):
        AllCombinations[:, i] = AllParams[i][idx[:, i]]
    # AllCombinations

    # saving
    lh_param_dict = {}
    ip = 0
    for (k,v) in zip(param_list_varied_cosmo.keys(),param_list_varied_cosmo.values()):
        lh_param_dict[v['class_name']] = AllCombinations[:,ip]
        ip+=1


    np.savez(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name+'/LHS_parameter_file.npz', **lh_param_dict)


    # saving
    for idx_process in range(n_processes):
        params = dict(zip(lh_param_dict.keys(), AllCombinations[idx_process*n_samples_per_process:(idx_process+1)*n_samples_per_process, :].T))
        np.savez(folder_path+'/LHS_parameter_file_{}.npz'.format(idx_process+1), **params)

def main():
    parser=argparse.ArgumentParser(description="generate spectra")
    parser.add_argument("-n_samples_per_process",help="n_samples_per_process" ,dest="n_samples_per_process", type=int, required=True)
    parser.add_argument("-n_processes",help="n_processes" ,dest="n_processes", type=int, required=True)
    parser.add_argument("-yaml_file",help="yaml_file" ,dest="yaml_file", type=str, required=True)
    parser.add_argument("-path_to_training_data_dir",help="path_to_training_data_dir" ,dest="path_to_training_data_dir", type=str, required=False)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
	main()
