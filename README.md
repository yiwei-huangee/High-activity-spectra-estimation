# Deep Learning for Energy Spectrum Estimation of High‑Activity Measurements

![workflow.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ef1e6330-020d-4bf2-b2d7-9ae4a9f006bc/74fcf482-0556-47c0-a376-32211a5c2cab/workflow.png)

This repository is for paper <Deep Learning for Energy Spectrum Estimation of High‑Activity Measurements>
Abstract: Estimating the spectrum from a high-activity source is a challenging problem, when the activity of the source is high, a physical phenomenon known as the pile-up effect distorts direct measurements, resulting in a significant bias to the standard estimators of the source spectrum used so far in the field. In this paper, we proposed a two-stage spectrum estimation method from radioactive sources with very high activity based on deep learning (DL), which combines the attention mechanism and convolutional neural network. Experiments show that this model can accurately estimate the spectrum under different energy distributions.

## Spectrum database

[spectrum database](https://github.com/OpenGammaProject/Gamma-Spectrum-Database)

## Gamma-simulator

The simulator we used in this paper can be found in (https://github.com/bykhov/gamma-simulator)

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

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ef1e6330-020d-4bf2-b2d7-9ae4a9f006bc/91d03356-f5a0-4779-a38b-598ae695ff56/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ef1e6330-020d-4bf2-b2d7-9ae4a9f006bc/8a7eaa34-022b-4838-a6c1-d9b1c42489a1/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ef1e6330-020d-4bf2-b2d7-9ae4a9f006bc/73ef1003-3154-4759-9f0b-abef6a94eba9/Untitled.png)

# Reference

https://github.com/Sakib1263/TF-1D-2D-Segmentation-End2EndPipelines/

https://github.com/OpenGammaProject/
