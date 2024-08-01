import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io as sio
import multiprocessing
from utils import *
from args import args
from models.UNetpp import Unetpp
from visualize import *
from gamma_simulator import gamma_simulator
from data_generate_with_GS import data_generate_with_GS
import os
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import csv
from sklearn.metrics import average_precision_score, precision_recall_curve
from collections import Counter
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, auc, roc_curve

# %% Main function
def main():
    if args.run_mode == 'train':
        print('Start training')

        '''using python simulator'''
        if args.mixture_lambda == False:
            simulator = gamma_simulator(verbose=True,
                                    verbose_plots=args.verbose_plots,
                                    source=args.source,
                                    lambda_value=args.train_lambda_n * args.fs,
                                    signal_len = args.train_sample*args.sample_length/args.fs,
                                    bins = args.bins,
                                    fs=args.fs,
                                    dict_type=args.dict_type,
                                    dict_shape_params=args.dict_shape_params,
                                    # dict_shape_params={'custom': True,
                                    #                    'param1bins': [2, 3],
                                    #                    'param1weights': [0.1, 0.2],
                                    #                    'param2bins': [0.4, 0.85],
                                    #                    'param2weights': [0.1, 0.2]},
                                    noise_unit=args.noise_unit,
                                    noise=args.noise,
                                    dict_size=args.dict_size,
                                    seed=args.train_seed)
            s = simulator.generate_signal()
            t = simulator.times
            e = simulator.energies
            std = simulator.noise_std
            source = simulator.energy_desc
            lambda_n = simulator.lambda_n

            if type(source) ==list:
                #convert list to str
                source = f'{source}'
                source = source.replace(' ', '')
            generate_asc(s,64,4096,std,source,lambda_n)
            '''cut off events exceed the signal length
            '''
            cut_index = np.where(t>args.sample_length*args.train_sample-0.5)
            t = np.delete(t,cut_index)
            e = np.delete(e,cut_index)
            times = np.round(t).astype(np.int32)
            Y_train = generate_label(times,e,args.train_sample)
            X_train = np.reshape(s,(args.train_sample,args.sample_length))
            Y_train = np.reshape(np.round(Y_train),(args.train_sample,args.sample_length))
        
        if args.mixture_lambda == True:
            s, t, e = data_generate_with_GS(args.train_lambda_value)
            X_train = []
            Y_train = []
            for lambda_n in args.train_lambda_value:
                times = np.round(t[f'{lambda_n}']).astype(np.int32)
                Y_train.append(generate_label(times,e[f'{lambda_n}'],500))
                X_train.append(s[f'{lambda_n}'])
            X_train = np.concatenate(X_train)
            Y_train = np.concatenate(Y_train)
            X_train = np.reshape(X_train,(args.train_sample,args.sample_length))
            Y_train = np.reshape(np.round(Y_train),(args.train_sample,args.sample_length))


        if args.model == 'Transformer':
            raw_inputs = remove_zeros(Y_train)
            padded_inputs = tf.keras.utils.pad_sequences(raw_inputs, padding="post")
            sos_token = tf.constant(1, dtype=tf.int32 ,shape=[2000, 1])
            eos_token = tf.constant(1, dtype=tf.int32, shape=[2000, 1])
            padded_inputs_sos = tf.concat([sos_token, padded_inputs, eos_token], axis=-1)
            Y_train = padded_inputs_sos.numpy()[:,1:-1]
            X_train = tf.concat([X_train,padded_inputs_sos.numpy()],axis=-1)
            input_size = X_train.shape[1]
        
        if args.problem_type == 'classification':
            Y_train_onehot = tf.one_hot(Y_train,depth=args.bins+1,axis=-1)
            Y_train = Y_train_onehot.numpy()
            
        """
        Generate the train dataset and validation dataset
        Proportion of training and validation is 8:2
        """
        
        train_idx, val_idx = getrandomIndex(X_train.shape[0],int(X_train.shape[0] * 0.8))
        Xtrain = X_train[train_idx]
        Xval = X_train[val_idx]
        ytrain= Y_train[train_idx]
        yval = Y_train[val_idx]
        train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
        val_dataset = tf.data.Dataset.from_tensor_slices((Xval, yval))

        train_dataset = train_dataset.shuffle(buffer_size=10*args.batch_size).batch(args.batch_size)
        val_dataset = val_dataset.shuffle(buffer_size=10*args.batch_size).batch(args.batch_size)


        """
        Define the model
        Using tensorflow MirroredStrategy to use multiple GPUs
        load MultiResUNet as the model
        """
        tf.keras.backend.clear_session()
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        Model = Unetpp(length=args.sample_length, model_depth=args.model_depth, num_channel=args.num_channel, 
                            model_width=args.model_width, kernel_size=args.kernel_size, problem_type=args.problem_type, 
                            output_nums=args.bins+1,ds=args.D_S, ae=args.A_E, ag=args.A_G, lstm=args.LSTM, 
                            alpha=args.alpha, feature_number = args.feature_number,is_transconv=args.is_transconv,MHA=args.MHA)
        

        # Model = transformer(input_size,num_channel,dim_val = 64, dim_attn= 64,n_heads=4,n_encoder_layers=6,n_decoder_layers=6,pe_len=1024)
        Model.compile(metrics='accuracy',loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=args.LR))
        callback_stp = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta = 1,
                                                    patience=20,
                                                    verbose=1,
                                                    restore_best_weights=True)
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        logger = tf.keras.callbacks.CSVLogger('/root/WorkSpace/project/spectrum_two_stage/results/log/training_lambda_'+
                        f'{lambda_n}'+'_SNR_'+f'{args.noise}'+'.log', separator=',', append=False)
        Model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[callback_reduce_lr,logger])#, callbacks=[callback_stp, callback_reduce_lr]
        Model.save_weights('/root/WorkSpace/project/spectrum_two_stage/results/h5/model.h5')
        # Model.summary()

    if args.run_mode == 'test':
        print('Start testing')


        simulator = gamma_simulator(verbose=True,
                                verbose_plots={'signal': True,'energy': True},
                                source=args.source,
                                lambda_value=args.test_lambda_n * args.fs,
                                signal_len = args.test_sample*args.sample_length/args.fs,
                                bins = args.bins,
                                fs=args.fs,
                                dict_type=args.dict_type,
                                dict_shape_params=args.dict_shape_params,
                                # dict_shape_params={'custom': True,
                                #                    'param1bins': [2, 3],
                                #                    'param1weights': [0.1, 0.2],
                                #                    'param2bins': [0.4, 0.85],
                                #                    'param2weights': [0.1, 0.2]},
                                noise_unit=args.noise_unit,
                                noise=args.noise,
                                dict_size=args.dict_size,
                                seed=args.test_seed)
        s = simulator.generate_signal()
        t = simulator.times
        e = simulator.energies
        source = simulator.energy_desc
        if type(source) ==list:
            #convert list to str
            source = f'{source}'
            source = source.replace(' ', '')
        train_hist_energy = simulator.train_hist_energy
        hist_energy = simulator.hist_energy
        lambda_n = round(simulator.lambda_n,2)

        # cut off events exceed the signal length
        cut_index = np.where(t > args.sample_length*args.test_sample-0.5)
        t = np.delete(t,cut_index)
        e = np.delete(e,cut_index)
        times = np.round(t).astype(np.int32)

        Y_test = generate_label(times,e,args.test_sample)
        X_test = np.reshape(s,(args.test_sample,args.sample_length)).astype(np.float32)
        Y_test = np.reshape(np.round(Y_test),(args.test_sample,args.sample_length)).astype(np.int16)


        """Define and load the model to test
        """
        Model = Unetpp(length=args.sample_length, model_depth=args.model_depth, num_channel=args.num_channel, 
                            model_width=args.model_width, kernel_size=args.kernel_size, problem_type=args.problem_type, 
                            output_nums=args.bins+1,ds=args.D_S, ae=args.A_E, ag=args.A_G, lstm=args.LSTM, 
                            alpha=args.alpha, feature_number = args.feature_number,is_transconv=args.is_transconv,MHA=args.MHA)
        

        # Model = transformer(dim_val = 32, dim_attn= 32,n_heads=4,n_encoder_layers=4,n_decoder_layers=3,pe_len=1024)
        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.LR), loss=tf.keras.losses.CategoricalCrossentropy(), metrics='accuracy')
        Model.load_weights('/root/WorkSpace/project/spectrum_two_stage/results/h5/model.h5')


        if args.problem_type == 'classification':
            Y_test_onehot = tf.one_hot(Y_test,depth=args.bins+1,axis=-1)
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_onehot)).batch(args.batch_size)
            Y_test_onehot_reshape = tf.reshape(Y_test_onehot,[args.test_sample*args.sample_length,args.bins+1])
            Y_test_onehot_reshape = Y_test_onehot_reshape.numpy()
            Model.evaluate(test_dataset)
            energy_pred1 = Model.predict(X_test[:2000,:])
            energy_pred1_argmax = tf.math.argmax(energy_pred1,axis=-1)
            energy_pred2 = Model.predict(X_test[2000:,:])
            energy_pred2_argmax = tf.math.argmax(energy_pred2,axis=-1)
            energy_pred_prob = np.concatenate((energy_pred1,energy_pred2),axis=0)
            energy_pred_prob = np.reshape(energy_pred_prob,[args.test_sample*args.sample_length,args.bins+1])
            energy_pred = np.concatenate((energy_pred1_argmax,energy_pred2_argmax),axis=0)
            energy = tf.reshape(energy_pred,[args.test_sample*args.sample_length]).numpy()
            
            '''plot estimated spectra'''
            hist_counts = np.bincount(energy)
            hist_counts = hist_counts[1:]
            hist_counts = np.pad(hist_counts,(0,args.bins-len(hist_counts)))
            norm_hist_counts = hist_counts/np.sum(hist_counts)
            ISE = calculate_ISE(simulator.hist_counts,norm_hist_counts)
            KL = calculate_KL(simulator.hist_counts,norm_hist_counts)
            print("ISE: ",ISE)
            print("KL",KL)
            plot_spectra(hist_energy, simulator.hist_counts, norm_hist_counts,source,lambda_n,args.bins,ISE,KL,'deepL')
            # save simulator.hist_counts to csv file if not exist
            if not os.path.exists('/root/WorkSpace/project/spectrum_two_stage/results/simulator_hist_counts_'+source+'.csv'):
                df = pd.DataFrame(simulator.hist_counts)
                df.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/simulator_hist_counts_'+source+'.csv',index=False)


            '''calculate MAE within the peaks'''
            if source == 'Co-60':
                # peaks: 1173.24 1332.50
                # max energy: 1665.23
                
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(1173.24/1665.23))-3:round(1024*(1173.24/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE1 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(1332.50/1665.23))-3:round(1024*(1332.50/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE2 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                if os.path.exists('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv'):
                    df1 = pd.read_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')
                    df2 = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','1173.24','1332.50'])
                    perf = pd.concat([df1,df2])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv',index=False, mode='w')
                else:
                    perf = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','1173.24','1332.50'])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')
            elif source == 'Cs-137':
                # peaks:31.80 661.66
                # max energy: 1665.23
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(31.80/1665.23))-3:round(1024*(31.80/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE1 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2

                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(661.66/1665.23))-3:round(1024*(661.66/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE2 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                if os.path.exists('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv'):
                    df1 = pd.read_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')
                    df2 = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','31.80','661.66'])
                    perf = pd.concat([df1,df2])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv',index=False, mode='w')
                else:
                    perf = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','31.80','661.66'])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')

            elif source == 'Co-57':
                # peaks:122.06 136.47
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(122.06/1665.23))-3:round(1024*(122.06/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                
                MAE1 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(136.47/1665.23))-3:round(1024*(136.47/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE2 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                if os.path.exists('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv'):
                    df1 = pd.read_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')
                    df2 = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','122.06','136.47'])
                    perf = pd.concat([df1,df2])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv',index=False, mode='w')
                else:
                    perf = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','122.06','136.47'])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')

            elif source == 'Na-22':
                # peaks:511.20 1274.54
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(511.20/1665.23))-3:round(1024*(511.20/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE1 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(1274.54/1665.23))-3:round(1024*(1274.54/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE2 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                if os.path.exists('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv'):
                    df1 = pd.read_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')
                    df2 = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','511.20','1274.54'])
                    perf = pd.concat([df1,df2])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv',index=False, mode='w')
                else:
                    perf = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','511.20','1274.54'])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')

            elif source == 'Cr-51':
                # peaks:0.17 319.69
                sorted_real_energy = np.sort(simulator.hist_counts[0:4])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                MAE1 = np.abs(peak_esitmated_1-peak_real_1)
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(319.69/1665.23))-3:round(1024*(319.69/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE2 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                if os.path.exists('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv'):
                    df1 = pd.read_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')
                    df2 = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','0.17','319.69'])
                    perf = pd.concat([df1,df2])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv',index=False, mode='w')
                else:
                    perf = pd.DataFrame([[lambda_n,MAE1,MAE2]],columns=[f'{lambda_n}','0.17','319.69'])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')

            elif source == 'Am-241':
                # peak: 59.54
                sorted_real_energy = np.sort(simulator.hist_counts[round(1024*(59.54/1665.23))-3:round(1024*(59.54/1665.23))+3])
                peak_real_1 = sorted_real_energy[-1]
                peak_real_2 = sorted_real_energy[-2]
                peak_real_1_idx = np.where(simulator.hist_counts==peak_real_1)
                peak_real_2_idx = np.where(simulator.hist_counts==peak_real_2)
                peak_esitmated_1 = norm_hist_counts[peak_real_1_idx[0][0]]
                peak_esitmated_2 = norm_hist_counts[peak_real_2_idx[0][0]]
                MAE1 = (np.abs(peak_esitmated_1-peak_real_1)+np.abs(peak_esitmated_2-peak_real_2))/2
                if os.path.exists('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv'):
                    df1 = pd.read_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')
                    df2 = pd.DataFrame([[lambda_n,MAE1]],columns=[f'{lambda_n}','59.54'])
                    perf = pd.concat([df1,df2])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv',index=False, mode='w')
                else:
                    perf = pd.DataFrame([[lambda_n,MAE1]],columns=[f'{lambda_n}','59.54'])
                    perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/MAE_'+source+'.csv')




            
            if source == 'Co-60':
                '''calculate micro average precision and recall'''
                precision, recall, average_precision = dict(), dict(), dict()
                # A "micro-average": quantifying score on all classes jointly
                precision["micro"], recall["micro"], _ = precision_recall_curve(
                    Y_test_onehot_reshape[:10240,:].ravel(), energy_pred_prob[:10240,:].ravel())
                print(precision["micro"].shape)
                average_precision["micro"] = average_precision_score(Y_test_onehot_reshape[:10240,:], energy_pred_prob[:10240,:], average="micro")
                sio.savemat('/root/WorkSpace/project/spectrum_two_stage/results/precision_recall_'+source+'_'+f'{lambda_n}'+'.mat', {'precision':precision, 'recall':recall, 'average_precision':average_precision})

                fpr, tpr, roc_auc = dict(), dict(), dict()
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_onehot_reshape[:10240,:].ravel(), energy_pred_prob[:10240,:].ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                sio.savemat('/root/WorkSpace/project/spectrum_two_stage/results/roc_auc_'+source+'_'+f'{lambda_n}'+'.mat', {'fpr':fpr, 'tpr':tpr, 'roc_auc':roc_auc})


                '''calculate confusion matrix
                ''' 
                cm = confusion_matrix(np.array(Y_test).reshape(-1), energy.reshape(-1),
                                      normalize=None, labels=np.arange(args.bins+1)+1)
    
                '''caclulate average precision, average recall and F1 score'''
    
                test_Precision = []
                for i in range(args.bins+1):
                    if sum(cm[:,i]) != 0:
                        test_Precision.append(cm[i,i]/sum(cm[:,i]))
                average_Precision = np.mean(test_Precision)
                print('avg_Precision', average_Precision)
    
                # calculate average recall
                test_Recall = []
                for i in range(args.bins+1):
                    if sum(cm[i,:]) != 0:
                        test_Recall.append(cm[i,i]/sum(cm[i,:]))
                average_Recall = np.mean(test_Recall)
                print('avg_Recall', average_Recall)
    
                F1 = 2*average_Precision*average_Recall/(average_Precision+average_Recall)
                print('F1', F1)
    
    
                '''ploting confusion matrix
                subplot 3 matrixs: 1-20 classes 100-120 classes 1000-1020 classes'''
    
                cm1_class = [100,150]
                cm2_class = [500,550]
                cm3_class = [960,1024]
                cm1 = cm[100:150,100:150]
                cm2 = cm[500:550,500:550]
                cm3 = cm[960:1024,960:1024]
    
                plt.figure(dpi=200)
                sns.heatmap(cm1,xticklabels=False, yticklabels=False, annot=False, cmap='crest_r')
                plt.xlabel('Predicted label:'+f'{cm1_class[0]}'+' to '+f'{cm1_class[1]}')
                plt.ylabel('True label')
                plt.title('Confusion Matrix')
                plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/CM/CM_'+source+'_'+
                                            f'{lambda_n}'+'_'+f'{cm1_class[0]}'+'to'+f'{cm1_class[1]}'+
                                            '_P_'+f'{average_Precision}'+'_R_'+f'{average_Recall}'+'_F1_'+f'{F1}'+'.png')
                plt.close()
    
                plt.figure(dpi=200)
                sns.heatmap(cm2,xticklabels=False, yticklabels=False, annot=False, cmap='crest_r')
                plt.xlabel('Predicted label:'+f'{cm2_class[0]}'+' to '+f'{cm2_class[1]}')
                plt.ylabel('True label')
                plt.title('Confusion Matrix')
                plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/CM/CM_'+source+'_'+
                                            f'{lambda_n}'+'_'+f'{cm2_class[0]}'+'to'+f'{cm2_class[1]}'+
                                            '_P_'+f'{average_Precision}'+'_R_'+f'{average_Recall}'+'_F1_'+f'{F1}'+'.png')
                plt.close()
    
                plt.figure(dpi=200)
                sns.heatmap(cm3,xticklabels=False, yticklabels=False, annot=False, cmap='crest_r')
                plt.xlabel('Predicted label:'+f'{cm3_class[0]}'+' to '+f'{cm3_class[1]}')
                plt.ylabel('True label')
                plt.title('Confusion Matrix')
                plt.savefig('/root/WorkSpace/project/spectrum_two_stage/results/CM/CM_'+source+'_'+
                                            f'{lambda_n}'+'_'+f'{cm3_class[0]}'+'to'+f'{cm3_class[1]}'+
                                            '_P_'+f'{average_Precision}'+'_R_'+f'{average_Recall}'+'_F1_'+f'{F1}'+'.png')
                plt.close()
            
            results = np.column_stack((simulator.hist_counts,hist_counts))
            np.savetxt('/root/WorkSpace/project/spectrum_two_stage/results/spectra_output.csv', 
                                            results, delimiter=',', header='real_spectra,spectra_pred', comments='')

        '''Caculate tom method's ISE and KL'''
        spectreDesempile = []
        spectreEmpile = []
        if os.path.exists('/root/WorkSpace/project/spectrum_two_stage/traditional_methods/tomcode/results/ds2Dfast_64_'+
                            source+'_'+f'{lambda_n}'+'.csv'):
            
            with open('/root/WorkSpace/project/spectrum_two_stage/traditional_methods/tomcode/results/ds2Dfast_64_'+
                                source+'_'+f'{lambda_n}'+'.csv', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for row in reader:
                    spectreDesempile.append(float(row[0]))
                    spectreEmpile.append(float(row[1]))
                spectreDesempile = np.array(spectreDesempile)
                spectreEmpile = np.array(spectreEmpile)
            KL_spectreDesempile = calculate_KL(simulator.hist_counts,spectreDesempile[0:args.bins])
            KL_spectreEmpile = calculate_KL(simulator.hist_counts,spectreEmpile[0:args.bins])
            ISE_spectreDesempile = calculate_ISE(simulator.hist_counts,spectreDesempile[0:args.bins])
            ISE_spectreEmpile = calculate_ISE(simulator.hist_counts,spectreEmpile[0:args.bins])
            plot_spectra(hist_energy, simulator.hist_counts, spectreDesempile[0:args.bins],source,lambda_n,args.bins,ISE_spectreDesempile,KL_spectreDesempile,'trad_Desempile')
            plot_spectra(hist_energy, simulator.hist_counts, spectreEmpile[0:args.bins],source,lambda_n,args.bins,ISE_spectreEmpile,KL_spectreEmpile,'trad_Empile')
            print(KL_spectreDesempile)
            print(KL_spectreEmpile)
            print(ISE_spectreDesempile)
            print(ISE_spectreEmpile)


        for i in range(1):
            perf = pd.DataFrame(columns=['label','prediction'])
            for j in range(energy_pred.shape[1]):
                perf.loc[j] = [Y_test[i*450,j],energy_pred[i*450,j]]
            perf.to_csv('/root/WorkSpace/project/spectrum_two_stage/results/discrepancy'+str(i*450)+'.csv')

if __name__ == '__main__':
    main()