import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
def plot_estimate(X,Y,i,loss):
    plt.figure(figsize=(12, 6))
    plt.plot(X, label='Energy Predictions')
    plt.plot(Y, label='Energy Labels')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Estimated Energy vs Labels'+'\n'+'lambda: '+''+str((i+1)*0.05)+'MAE: '+str(loss))
    plt.legend()
    plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/test_{}.png'.format(i*450), format='png')

def histogram(energy,real_spectrum):
    spectrum = np.zeros((1024,1))
    for i in range(energy.shape[0]):
        if energy[i]!=0:
            spectrum[round(energy[i])] =spectrum[round(energy[i])]+1
            i = i+1
    # normalize
    spectrum = spectrum/np.sum(spectrum)
    plt.figure(figsize=(12, 6))
    plt.plot(real_spectrum, label='Energy Labels')
    plt.plot(spectrum, label='Energy Predictions')
    
    plt.xlabel('Energy')
    plt.ylabel('Counts')
    plt.title('Estimated spectrum')
    plt.legend()
    plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/sepctrum.png', format='png')
    return spectrum

def ConfusionMatrix(cm):
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/confusion_matrix.pdf')
    plt.close()

def plot_spectra(hist_energy, real_hist_counts, hist_counts,source,lambda_n,bins,ISE,KL,method):
    """Plot the energy histogram"""
    plt.bar(hist_energy, real_hist_counts, width=2, alpha=0.5, label='Reference spectrum(label)')
    plt.bar(hist_energy, hist_counts, width=2, alpha=0.5, label='Estimated spectrum')
    plt.yscale('log')
    plt.grid(linestyle='--', linewidth=1, color='gray')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Normalized counts')
    
    # plt.title(f'energy spectrum estimation under {lambda_n} with {hist_energy.shape[0]} bins')
    if type(source) ==list:
        #convert list to str
        source = f'{source}'
    plt.legend()
    plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/figs/'+method+'_'+'lambda_'+
                        f'{round(lambda_n,4)}'+'_'+source+'_bins_'+f'{bins}'+'_'+f'{ISE}'+'_'+f'{KL}'+'.png', format='png')
    plt.close()