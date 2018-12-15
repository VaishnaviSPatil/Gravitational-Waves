import numpy as np
import matplotlib.pyplot as plt
from bbhSignal import genBBHSignal1
from bbhSignalUtils import getPSD, genRandMassPairs
import h5py
#-- Define some constants
tbuffer = 1.50    
fLow = 30.0
fSamp = 8192.0
N = int(tbuffer*fSamp)

waveModel = 'SEOBNRv4'

#-- Generate the PSD
psd = getPSD(fLow, 1./tbuffer, N)

#-- Seed the random number generator
np.random.seed(2017)

#-- Range of BBH parameters and SNR
m1_min, m1_max = 12.5, 35.
m2_min, m2_max = 12.5, 35.
kMax_train = 120
m1_tr, m2_tr = genRandMassPairs(m1_min, m1_max, m2_min, m2_max, kMax_train)
kMax_test = 30
m1_te, m2_te = genRandMassPairs(m1_min, m1_max, m2_min, m2_max, kMax_test)
mass_data_train =  np.stack((np.asarray(m1_tr),np.asarray(m2_tr)),axis=0)
mass_data_test =  np.stack((np.asarray(m1_te),np.asarray(m2_te)),axis=0)
# with h5py.File('masses.h5', 'w') as hf:
#     hf.create_dataset("train",  data=mass_data_train)
#     hf.create_dataset("test",  data=mass_data_test)

# -- Noisy Versions
nEpochs = 5


rho_list = np.concatenate((np.arange(0,10,2),np.arange(10,130,10)),axis=0)
for i in range(len(rho_list)-1):
    tr_data=np.empty(shape=[0,3,1344])
    tr_params=np.empty(shape=[0,3])
    for k in range(kMax_train):
        rho_target = np.random.uniform(rho_list[i],rho_list[i+1],1)
        params = np.asarray([m1_tr[k],m1_tr[k],rho_target]*nEpochs).T.reshape([nEpochs,3])
        tr_params = np.vstack((tr_params,params))
        st, pLoc, nt = genBBHSignal1(fLow, fSamp, tbuffer, waveModel, m1_tr[k], m2_tr[k], 0., 0., rho_target, psd, nEpochs)
        dtmp = np.stack((np.asarray(st),np.asarray(st)+np.asarray(nt),np.asarray(nt)),axis=1)
        tr_data = np.vstack((tr_data,dtmp))
        if(k%30==0):
            print("Train ==> rho:{} k:{}".format(rho_list[i],k))
            print(tr_data.shape)
            print(tr_params.shape)
    te_data=np.empty(shape=[0,3,1344])
    te_params=np.empty(shape=[0,3])
    for k in range(kMax_test):
        rho_target = np.random.uniform(rho_list[i],rho_list[i+1],1)
        params = np.asarray([m1_te[k],m1_te[k],rho_target]*nEpochs).T.reshape([nEpochs,3])
        te_params = np.vstack((te_params,params))
        st, pLoc, nt = genBBHSignal1(fLow, fSamp, tbuffer, waveModel, m1_te[k], m2_te[k], 0., 0., rho_target, psd, nEpochs)
        dtmp = np.stack((np.asarray(st),np.asarray(st)+np.asarray(nt),np.asarray(nt)),axis=1)
        te_data = np.vstack((te_data,dtmp))
        if(k%10==0):
            print("Test ==> rho:{} k:{}".format(rho_list[i],k))
            print(te_data.shape)
            print(te_params.shape)
    with h5py.File('rho_{0:0=3d}_{1:0=3d}.h5'.format(rho_list[i],rho_list[i+1]), 'w') as hf:
        hf.create_dataset("train_data",  data=tr_data)
        hf.create_dataset("train_params",  data=tr_params)
        hf.create_dataset("test_data",  data=te_data)
        hf.create_dataset("test_params",  data=te_params)
