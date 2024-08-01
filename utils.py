from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from args import args
import random
import h5py
import math
from sklearn.metrics import hamming_loss
# from tensorflow.keras.utils import to_categorical

# %% Data loading (using true arrival time)
def load_train_data(args):
    """Load data from .mat file
    return: X_train, Y_train, sample_length
    """
    mat = loadmat(args.data_directory + 'train_1.mat')
    # mat = loadmat(data_directory + 'train_len_512.mat')
    # X_train = mat['X_train_all'] / 100. -0.5 # normalization
    X_train = mat['X_train_all']
    Y_train = mat['energy_label_all']
    # if args.model == 'Transformer':
        
    #     X_train = tf.concat([X_train, Y_train], axis=-1).numpy()
    sample_length = X_train.shape[-1]  # frame length
    return X_train, Y_train,sample_length



def load_test_data(args):
    """Load data from .mat file
    return: X_train, Y_train, sample_length
    """
    # init 3D matrix
    # X_train_all = np.zeros((train_sample,args.N,1024))
    # Y_train_all = np.zeros((train_sample,1638))
    
    mat = loadmat(args.data_directory + 'test_1.mat')
    # mat = loadmat(data_directory + 'train_len_512.mat')
    # X_train = mat['X_train_all'] / 100. -0.5 # normalization
    X_test = mat['X_test_all']
    
    # X_train = X_train / 7472 #min-max normalization
    # z-score normalization
    # for i in range(X_train.shape[0]):
    #     mu = np.mean(X_train[i,:])
    #     sigma = np.std(X_train[i,:])
    #     for j in range(X_train.shape[1]):
    #         X_train[i,j] = (X_train[i,j] - mu) / sigma
    # Y_train = to_categorical(mat['Y_train_all'], num_classes=num_classes)
    Y_test = mat['test_energy_label_all']
    sample_length = X_test.shape[-1]  # frame length
    train_spectrum = mat['train_spectrum']
    # plt.figure(figsize=(12, 6))
    # plt.plot(train_spectrum, label='Energy label')
    # plt.xlabel('Energy')
    # plt.ylabel('probability')
    # plt.title('Real spectrum')
    # plt.legend()
    # plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/real_epctrum.png', format='png')
    return X_test, Y_test,train_spectrum,sample_length



# def load_test_data(j: int,k: int, data_directory='/root/WorkSpace/project/spectrum_two_stage/database/datatest/', num_classes=6):
#     """Load data from .mat file
#     k: 10, 20, 30, 40, 50, 60
#     returns: X_test, Y_test, Y_test_i
#     """
#     mat = loadmat(data_directory + str(j)+ '_data_test_' + str(k) +'.mat')
#     # mat = loadmat(data_directory + 'data_test_' + str(k)+'_len_512' +'.mat')
#     X_test = mat['X_test'] / 100. -0.5
#     Y_test = mat['Y_test']
#     # Y_test = to_categorical(Y_test_i, num_classes=num_classes)
#     # Y_test = mat['Y_test'].astype(bool).astype(np.uint8)
#     return X_test, Y_test


def save_test_data(k: int, Y_pred, data_directory='/root/WorkSpace/project/activity/database/'):
    """Save test data to .mat file"""
    file_name = os.getcwd() + '\\data_pred_' + str(k) + '.mat'
    savemat(file_name, {'Y_pred': Y_pred})

# %% Functions
def generate_label_(times,e,train_sample):
    Y_train = np.zeros(1024*train_sample)
    for i in range(len(times)):
        Y_train[times[i]:times[i]+4] = e[i]
    return Y_train


def generate_label(times,e,train_sample):
    """Generate label for training: The energy of the events are distributed according to the event times
    t:[...,19,19,19,...] e:[...,138,125,260,...] Y_train:[...   ...]
    up to 6 events at the same time
    """
    Y_train = np.zeros(args.sample_length*train_sample)
    for i in range(len(times)):
        if Y_train[times[i]] == 0:
            Y_train[times[i]] = e[i]
        else:
            if Y_train[times[i]+1] == 0:
                Y_train[times[i]+1] = e[i]
            else:
                if Y_train[times[i]+2] == 0:
                    Y_train[times[i]+2] = e[i]
                else:
                    if Y_train[times[i]+3] == 0:
                        Y_train[times[i]+3] = e[i]
                    else:
                        if Y_train[times[i]+4] == 0:
                            Y_train[times[i]+4] = e[i]
                        else:
                            if Y_train[times[i]+5] == 0:
                                Y_train[times[i]+5] = e[i]
                            else:
                                if Y_train[times[i]+6] == 0:
                                    Y_train[times[i]+6] = e[i]
                                else:
                                    continue
    return Y_train


def DiceLoss_CrossEntropy(y_true, y_pred, n_classes=args.bins+1):
    diceloss = DiceLoss(n_classes)
    cce = tf.keras.losses.CategoricalCrossentropy()
    ce_loss = cce(y_true, y_pred)
    y_pred = tf.one_hot(tf.argmax(y_pred,axis=-1),1025)
    d_loss = diceloss(y_true,y_pred)
    loss = 0.8 * ce_loss+ 0.2 * d_loss
    return loss
class DiceLoss(tf.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, target,score):
        smooth = 1e-10
        intersect = tf.reduce_sum(score * target)
        y_sum = tf.reduce_sum(target * target)
        z_sum = tf.reduce_sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def __call__(self,target, inputs):
        dice = self._dice_loss(inputs, target)
        return dice / self.n_classes

def Cross_entropy_hamming_loss(y_true,y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    ce_loss = cce(y_true,y_pred)
    y_pred = tf.one_hot(tf.argmax(y_pred,axis=-1),1025)
    xor = tf.cast(y_true,dtype=tf.int32) ^ tf.cast(y_pred,dtype=tf.int32)
    h_loss = tf.reduce_sum(xor) / (1024 * 1025)
    return 0.8 * ce_loss + 0.2 * h_loss

def MSE_KL_loss(y_true, y_pred,weight=1.0):
    """MSE + KL loss
    """
    # MSE loss
    mse_loss = tf.keras.losses.MSE(y_true, y_pred)
    # KL loss
    kl_loss = tf.keras.losses.KLDivergence()(y_true, y_pred)
    return mse_loss + weight*kl_loss

def getrandomIndex(n,x):
    """Get random index for training and validation
    """
    train_idx = random.sample(range(n),x)
    val_idx = list(set(range(n))-set(train_idx))
    return train_idx,val_idx

def remove_zeros(matrix):
    new_matrix = [[element for element in row if element != 0] for row in matrix]
    return new_matrix

def calculate_ISE(fx,fy):
    ISE = 0
    for i in range(len(fy)):
        ISE = ISE + (fx[i]-fy[i])*(fx[i]-fy[i])
    return '{:e}'.format(ISE)

def calculate_KL(fx,fy):
    KL = 0
    for i in range(len(fy)):
        if fx[i] <= 0:
            fx[i] = 1e-8
        if fy[i] <= 0:
            fy[i] = 1e-8
        KL = KL + (fx[i] * math.log(fx[i]/fy[i])) + (fy[i] * math.log(fy[i]/fx[i]))
    return '{:e}'.format(KL / len(fx))

def generate_asc(s,duration,bins,std,source,lambda_n):
    '''This function replace Tom's code to generate the 3D density asc file
    Using python simulator
    '''
    l,r,i = 0,0,0
    energies = np.zeros((duration,bins))
    while i < len(s):
        if s[i] > 3*std:
            l,r = i,i
            while s[r] > 3*std and r < len(s): 
                r += 1
            if round(sum(s[l:r]))<bins and r-l < duration:
                energies[r-l,round(sum(s[l:r]))] += 1
            i = r
        else:
            i+=1

    energies = np.ravel(energies)
    cdenergies= np.cumsum(energies)/sum(energies)
    f=open('/root/WorkSpace/project/spectrum_two_stage/traditional_methods/tomcode/acsfiles/densite_3d_pour_simulation'+
                        f'{bins}'+'_'+f'{duration}'+'_'+source+'_'+f'{lambda_n}'+'.asc', 'wt')
    f.write(str(bins)+'\n')
    f.write(str(duration)+'\n')
    for val in cdenergies:	f.write(str(val)+'\n')
    f.close()
    return 0

def calculate_MAE(peak1,peak2,norm_hist_counts,simulator_hist_counts):
    sorted_estimated_energy = np.sort(norm_hist_counts[round(1024*(peak1/1665.23))-3:round(1024*(peak1/1665.23))+3])
    peak_esitmated_1 = sorted_estimated_energy[-1]
    peak_esitmated_2 = sorted_estimated_energy[-2]
    sorted_real_energy = np.sort(simulator_hist_counts[round(1024*(peak1/1665.23))-3:round(1024*(peak1/1665.23))+3])
    peak_real_1 = sorted_real_energy[-1]
    peak_real_2 = sorted_real_energy[-2]
    MAE1 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
    sorted_estimated_energy = np.sort(norm_hist_counts[round(1024*(peak2/1665.23))-3:round(1024*(peak2/1665.23))+3])
    peak_esitmated_1 = sorted_estimated_energy[-1]
    peak_esitmated_2 = sorted_estimated_energy[-2]
    sorted_real_energy = np.sort(simulator_hist_counts[round(1024*(peak2/1665.23))-3:round(1024*(peak2/1665.23))+3])
    peak_real_1 = sorted_real_energy[-1]
    peak_real_2 = sorted_real_energy[-2]
    MAE2 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
    return MAE1,MAE2