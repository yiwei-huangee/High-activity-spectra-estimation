# Deep Learning for Energy Spectrum Estimation of High‑Activity Measurements
![workflow](https://github.com/user-attachments/assets/1043eec4-0f14-49f5-b1af-5d0eaefcee7d)

This repository is for paper <Deep Learning for Energy Spectrum Estimation of High‑Activity Measurements>
Abstract: Estimating the spectrum from a high-activity source is a challenging problem, when the activity of the source is high, a physical phenomenon known as the pile-up effect distorts direct measurements, resulting in a significant bias to the standard estimators of the source spectrum used so far in the field. In this paper, we proposed a two-stage spectrum estimation method from radioactive sources with very high activity based on deep learning (DL), which combines the attention mechanism and convolutional neural network. Experiments show that this model can accurately estimate the spectrum under different energy distributions.

## Spectrum database

[spectrum database](https://github.com/OpenGammaProject/Gamma-Spectrum-Database)

## Gamma-simulator

The simulator we used in this paper can be found at (https://github.com/bykhov/gamma-simulator)

## Train

You can train the model with

```python
python main.py --run_mode='train' --source=[source_name] --bins=1024 --batch_size=16 --train_lambda_n=[lambda value] --fs=[sampling rate] \
--train_sample=[train samples] --noise=[noise] --dict_size=[shape size number] --train_seed=[train seed] \
--model_depth=[model depth] --model_width=[model width] --n_heads=[number heads]
```

or use Makefile

```makefile
train_Ac-225_0.08:
	$(call RUN_TRAIN_WORK,Ac-225,0.08)
train_Am-241_0.08:
	$(call RUN_TRAIN_WORK,Am-241,0.08)
train: train_Ac-225_0.08 train_Am-241_0.08
```

```python
make train
```

## Test

```python
python main.py --run_mode='test'  --bins=1024 --source=[source_name] --batch_size=16 --test_lambda_n=[test lambdaa]
```

```makefile
test_Ac-225_0.08:
	$(call RUN_TEST_WORK2,Ac-225,0.08)
test_Am-241_0.08:
	$(call RUN_TEST_WORK2,Am-241,0.08)
test: test_Ac-225_0.08 test_Am-241_0.08
```

```makefile
make test
```
## Experiment results
The estimated spectrum of Ac-225 ($\lambda=0.08$) based on Deep learning(a), fast pile-up correction(b), and no pile-up correction(c)

![deepL_lambda_0 08_Ac-225_bins_1024_8 971090e-05_9 264578e-05](https://github.com/user-attachments/assets/432f7c08-7df5-4841-b5bc-13181902a02c)
(a)
![trad_Desempile_lambda_0 08_Ac-225_bins_1024_2 090786e-02_1 342669e-02](https://github.com/user-attachments/assets/e9e63278-0ad3-43c5-ba00-8d0455cefcfc)
(b)
![trad_Empile_lambda_0 08_Ac-225_bins_1024_1 415298e-02_1 100465e-02](https://github.com/user-attachments/assets/62685424-2998-43e8-8972-516eb31d8264)
(c)[deepL_lambda_0.08_Ac-225_bins_1024_1.304901e-04_1.329449e-04.pdf](https://github.com/user-attachments/files/17374950/deepL_lambda_0.08_Ac-225_bins_1024_1.304901e-04_1.329449e-04.pdf)

Estimated spectrum based on Deep leaning, signal source as the example.


[deepL_lambda_0.08_Cs-137_bins_1024_2.283977e-04_4.151637e-05.pdf](https://github.com/user-attachments/files/17374955/deepL_lambda_0.08_Cs-137_bins_1024_2.283977e-04_4.151637e-05.pdf)
[deepL_lambda_0.08_Co-60_bins_1024_7.187970e-05_5.314916e-05.pdf](https://github.com/user-attachments/files/17374954/deepL_lambda_0.08_Co-60_bins_1024_7.187970e-05_5.314916e-05.pdf)
[deepL_lambda_0.08_Ba-131_bins_1024_2.612169e-04_5.930216e-05.pdf](https://github.com/user-attachments/files/17374953/deepL_lambda_0.08_Ba-131_bins_1024_2.612169e-04_5.930216e-05.pdf)
[deepL_lambda_0.08_Am-241_bins_1024_7.920451e-04_9.538854e-05.pdf](https://github.com/user-attachments/files/17374951/deepL_lambda_0.08_Am-241_bins_1024_7.920451e-04_9.538854e-05.pdf)
[Uploading deepL_lambda_0.08_Ac-225_bins_1024_1.304901e-04_1.329449e-04.pdf…]()
[deepL_lambda_0.08_I-125_bins_1024_1.600681e-04_1.995330e-04.pdf](https://github.com/user-attachments/files/17374948/deepL_lambda_0.08_I-125_bins_1024_1.600681e-04_1.995330e-04.pdf)


# Reference

https://github.com/Sakib1263/TF-1D-2D-Segmentation-End2EndPipelines/

https://github.com/OpenGammaProject/
