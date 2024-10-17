# Deep Learning for Energy Spectrum Estimation of High‑Activity Measurements

![workflow](https://github.com/user-attachments/assets/e8d90893-0416-4b88-9cc4-511b026ec8bd)

The energy spectrum of a high-activity gamma source is hard to estimate due to the pileup effect, and solutions to this challenging problem are sought in a number of application areas. In this paper, we proposed a solution based on deep learning (DL), that combines the self-attention mechanism and convolutional neural network (CNN) architectures. The performance analysis was conducted using spectral data from a small scintillator (NaI) and is based on a dedicated simulator, and the results indicate that our model can accurately predict energies even at high count rates. The probability distribution distance and the energy resolution metrics of the predicted energy histograms are small, which is important for subsequent peak analysis and source identification. We demonstrate that our proposed method remains robust and accurate in predicting spectra under high to very high activities, even with varying sources and noise intensities.

## Spectrum Database

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
## Network Architecture
<figure>
  <img src="https://github.com/user-attachments/assets/368b1070-097f-48e2-9b67-b6917cb58283" alt="Network Architecture" width="100%">
  <figcaption>Illustration of our proposed neural network architecture \textit{AttnUNet++}.</figcaption>
</figure>


## Experiment Results

### Estimated spectrum based on Deep learning, signal source as the example.
<table>
  <tr>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/4ca84184-fac3-4681-b894-bb0737a66add" alt="I-125" width="100%"/>
        <figcaption>(a) I-125, $ISE=14.68 \times 10^{-5}$</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/2d403f3d-2d20-4048-aeae-fa05eb851b9e" alt="Cs-137" width="100%"/>
        <figcaption>(b) Cs-137, $ISE=15.18 \times10^{-5}$</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/4b380f8c-6070-4d9a-9ffa-acdaf1152c7f" alt="Co-60" width="100%"/>
        <figcaption>(c) Co-60, $ISE=10.54 \times 10^{-5}$</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/dfefdc94-1cdf-43b1-aad5-3712fb6e7b4c" alt="Ba-131" width="100%"/>
        <figcaption>(d) Ba-131, $ISE=18.50 \times 10^{-5}$</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/0a988a2e-08e4-4480-a899-dab3817325ca" alt="Am-241" width="100%"/>
        <figcaption>(e) Am-241, $ISE=57.15 \times 10^{-5}$</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/d8ad4eaf-dab4-4d24-87c5-2157d8f58809" alt="Ac-225" width="100%"/>
        <figcaption>(f) Ac-225, $ISE=11.64 \times 10^{-5}$</figcaption>
      </figure>
    </td>
  </tr>
</table>

### Comparison between DL and traditional methods
<table>
  <tr>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/432f7c08-7df5-4841-b5bc-13181902a02c" alt="Image 1" width="100%">
        <figcaption>(a) Estimated spectrum of Ac-225 ($\lambda=0.08$) based on Deep learning</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/e9e63278-0ad3-43c5-ba00-8d0455cefcfc" alt="Image 2" width="100%">
        <figcaption>(b) Estimated spectrum of Ac-225 ($\lambda=0.08$) based on Fast pile-up Correction</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/user-attachments/assets/62685424-2998-43e8-8972-516eb31d8264" alt="Image 3" width="100%">
        <figcaption>(c) Estimated spectrum of Ac-225 ($\lambda=0.08$) with no pile-up correction(c)</figcaption>
      </figure>
    </td>
  </tr>
</table>

### Probability Distribution Distance
<figure>
  <img src="https://github.com/user-attachments/assets/d6e6f0a6-30c0-4d95-bb70-0b0e8c1c7a14" alt="ISEandKL" width="100%">
</figure>

<figure>
  <img src="https://github.com/user-attachments/assets/2ee8b2cb-6e3c-46a9-83de-721cfa02f0e6" alt="MAEs" width="100%">
</figure>

# Reference

https://github.com/Sakib1263/TF-1D-2D-Segmentation-End2EndPipelines/

https://github.com/OpenGammaProject/
