from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d
import sys
import emcee
import h5py as h5
from integrate_emulator_predictions import integrate_emulator_predictions
from NN_predict_multi import *
import pickle
import xarray as xr

import pickle

def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')
                

def setup(options):

    path_chains =  options.get_string(option_section, "path_chains", default = "")
    key =  options.get_string(option_section, "key", default = "")

    moments_conf = load_obj(path_chains)
    info = moments_conf[key]

    config = dict()
    #config["Sellentin"] = info["Sellentin"]

    inv_cov,y_obs= info['inv_cov'],info['y_obs']
    #xx,inv_cov,scales,bins,emu_setup,y_obs,tm,dc= info['Nz'],info['inv_cov'],info['scales'],info['bins'],info['emu_config'],info['y_obs'],info["transf_matrix"],info["dc"]
    #config["xx"] =  xx
    config["inv_cov"] =  inv_cov
    #config["scales"] =  scales
    #config["bins"] =  bins
    #config["emu_setup"] =  emu_setup
    config["y_obs"] =  y_obs
    #config["tm"] =  tm
    #config["dc"] =  dc
    config["cov"] =  info['cov']

    rescale_features_filename = "rescale_feature_file.pkl"
    model_filename = "emulator_25_Jul_MODEL_1e5"
    with open(rescale_features_filename, 'rb') as file:
        rescale_features = pickle.load(file)

    mini = rescale_features['small']
    maxi = rescale_features['large']

    modes = [0,1,2,3,4,5,6,7]
    train_params = np.load('training_params_normalized_1e5.npz')
    model_parameters = train_params.files
    cp_nn = cosmopower_NN(parameters=model_parameters,
                      modes=modes,
                      #n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                      #n_hidden= [256, 256, 256, 256], # 4 hidden layers, each with 256 nodes
                      #n_hidden = [512, 512, 1024, 512, 512], #5 hidden layers, each with 512 nodes
                      #n_hidden = [512, 1024, 1024, 512],
                      #n_hidden = [128, 512, 1024, 512, 128], #GOOD
                      #n_hidden = [128, 512, 1024, 1024, 512, 128], #VERY GOOD
                      n_hidden = [128, 512, 1024, 1024, 1024, 512, 128],
                      verbose=True, # useful to understand the different steps in initialisation and training
                     )

    cp_nn.restore(model_filename)

    emulator_model = dict()

    emulator_model['mini'] = mini
    emulator_model['maxi'] = maxi
    emulator_model['cp_nn'] = cp_nn

    config['emulator_model'] = emulator_model
            
    return config

def execute(block, config):

    name_likelihood = 'shear_3PCF_like'

    k_max = 50

    params = dict()
    #params['Omega_m'] = block['cosmological_parameters','omega_m']
    #params['Sigma_8'] = block['cosmological_parameters','sigma_8']
    #params['sigma8_input'] = block['cosmological_parameters','sigma8_input']
    params['n_s'] = block['cosmological_parameters','n_s']
    params['Omega_b'] = block['cosmological_parameters','omega_b']
    params['h'] = block['cosmological_parameters','h0']
    params['A_s'] = block['cosmological_parameters','As']
    params['w0_fld'] = block['cosmological_parameters','w']
    params['Omega_cdm'] = block['cosmological_parameters','omega_m']-params['Omega_b']
    params['Omega_Lambda'] = 1-block['cosmological_parameters','omega_m']
    params['output'] = 'mPk'
    params['non linear'] = 'halofit'
    params['P_k_max_1/Mpc'] = k_max * params['h']
    params['z_max_pk'] = 10

    # read Nz and interpolate it ****************************************************************

    #nbin = block['nz_source', "nbin"]
    nbin = 4
    z = block['nz_source', "z"]
    Nz0 = []
    for i in range(1, nbin + 1):
        nz = block['nz_source', "bin_%d" % i]
        #fz = interp1d(z,nz)
        fz = np.ndarray(shape=(len(nz),2))
        fz[:,0] = z
        fz[:,1] = nz
        Nz0.append(fz)
    config["Nz0"] = Nz0

    # m and dz must be re-collected into 1 array.
    #m = []
    #dz = []
    #for i in range(len(Nz0)):
    #    m.append(block['shear_calibration_parameters','m{0}'.format(i+1)])
    #    dz.append(0.)
    #params['m'] = m
    #params['dz'] = dz

    zarray = np.linspace(0.05,2.2,55)

    triangles_file = np.load("Sep11_8_anglebins_12_r_bins_2000_patches_test_triangleshapes.npy")

    triangle_configs = [triangles_file[:96], triangles_file[96:192], triangles_file[192:288], triangles_file[288:]]

    emulator_model = config['emulator_model']
    cp_nn = emulator_model['cp_nn']
    mini = emulator_model['mini']
    maxi = emulator_model['maxi']

    predictions = predict(params, zarray, "rescale_parameters_file.pkl", triangle_configs, cp_nn, mini, maxi)

    integrated_predictions = []

    integrated_predictions_bin_1 = integrate_emulator_predictions(params, zarray, predictions[0], config['Nz0'][0])
    integrated_predictions_bin_2 = integrate_emulator_predictions(params, zarray, predictions[1], config['Nz0'][1])
    integrated_predictions_bin_3 = integrate_emulator_predictions(params, zarray, predictions[2], config['Nz0'][2])
    integrated_predictions_bin_4 = integrate_emulator_predictions(params, zarray, predictions[3], config['Nz0'][3])
    integrated_predictions.append(integrated_predictions_bin_1)
    integrated_predictions.append(integrated_predictions_bin_2)
    integrated_predictions.append(integrated_predictions_bin_3)
    integrated_predictions.append(integrated_predictions_bin_4)

    y = np.ndarray(shape=(4*len(integrated_predictions[0])*len(Nz0)), dtype=complex)
    for nbinn in range(len(Nz0)):
        yy = np.ndarray(shape=(4*len(integrated_predictions[nbinn])), dtype=complex)
        yy[:len(integrated_predictions[nbinn])] = integrated_predictions[nbinn][:,0] + 1j*integrated_predictions[nbinn][:,1]
        yy[len(integrated_predictions[nbinn]):2*len(integrated_predictions[nbinn])] = integrated_predictions[nbinn][:, 2]\
                                                                                      + 1j * integrated_predictions[nbinn][:, 3]
        yy[2*len(integrated_predictions[nbinn]):3*len(integrated_predictions[nbinn])] = integrated_predictions[nbinn][:, 4] \
                                                                                        + 1j * integrated_predictions[nbinn][:, 5]
        yy[3*len(integrated_predictions[nbinn]):] = integrated_predictions[nbinn][:, 6] + 1j * integrated_predictions[nbinn][:, 7]

        y[nbinn*4*len(integrated_predictions[nbinn]):(nbinn+1)*4*len(integrated_predictions[nbinn])] = yy

    print(np.shape(config['y_obs']))
    print(np.shape(y))
    w = y-config['y_obs']
    #to_file = np.ndarray(shape=(3,len(config['y_obs'])))
    #to_file[0] = np.real(y)
    #to_file[1] = np.real(config['y_obs'])
    #to_file[2] = np.sqrt(np.real(np.diag(config['cov'])))
    #np.save("test_sampler_theory_and_data_6Oct_2", to_file)
    #print(params)
    np.save("theory_cosmogridparams_Dec7",y)

    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))
    block[names.likelihoods, name_likelihood] = -0.5 * np.real(chi2)

    return 0

def cleanup(config):
    pass
