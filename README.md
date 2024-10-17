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

### Estimated spectrum based on Deep learning, signal source as the example.
<table>
  <tr>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/4ca84184-fac3-4681-b894-bb0737a66add" width="300"/>
        <figcaption>Caption 1: Description of image 1</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/2d403f3d-2d20-4048-aeae-fa05eb851b9e" width="300"/>
        <figcaption>Caption 2: Cs-137,ISE=15.18x10^-5</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/4b380f8c-6070-4d9a-9ffa-acdaf1152c7f" width="300"/>
        <figcaption>Caption 3: Co-60,ISE</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/dfefdc94-1cdf-43b1-aad5-3712fb6e7b4c" width="300"/>
        <figcaption>Caption 4: Description of image 4</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/0a988a2e-08e4-4480-a899-dab3817325ca" width="300"/>
        <figcaption>Caption 5: Description of image 5</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/d8ad4eaf-dab4-4d24-87c5-2157d8f58809" width="300"/>
        <figcaption>Caption 6: Description of image 6</figcaption>
      </figure>
    </td>
  </tr>
</table>



The estimated spectrum of Ac-225 ($\lambda=0.08$) based on Deep learning(a), fast pile-up correction(b), and no pile-up correction(c)

![deepL_lambda_0 08_Ac-225_bins_1024_8 971090e-05_9 264578e-05](https://github.com/user-attachments/assets/432f7c08-7df5-4841-b5bc-13181902a02c)
(a)
![trad_Desempile_lambda_0 08_Ac-225_bins_1024_2 090786e-02_1 342669e-02](https://github.com/user-attachments/assets/e9e63278-0ad3-43c5-ba00-8d0455cefcfc)
(b)
![trad_Empile_lambda_0 08_Ac-225_bins_1024_1 415298e-02_1 100465e-02](https://github.com/user-attachments/assets/62685424-2998-43e8-8972-516eb31d8264)
(c)






# Reference

https://github.com/Sakib1263/TF-1D-2D-Segmentation-End2EndPipelines/

https://github.com/OpenGammaProject/
