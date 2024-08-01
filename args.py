import argparse
import ast
parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='train')
parser.add_argument('--using_arival_time', type=str, default='real')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--data_directory', type=str,
    default='/root/WorkSpace/project/spectrum_two_stage/database/')

'''Train simulation parameters'''
parser.add_argument('--mixture_lambda', type=bool, default=False)
parser.add_argument('--source', type=ast.literal_eval,default="{'name':'Co-60','weights':1}")
parser.add_argument('--verbose_plots', type=dict,
                            default={'signal': True,'energy': True})
parser.add_argument('--fs', type=int, default=1e6)
parser.add_argument('--train_lambda_n', type=float or list, default=0.06)
parser.add_argument('--train_sample', type=int, default=7000)
# parser.add_argument('--signal_len', type=int, default=3.072)
parser.add_argument('--noise', type=float, default=70)
parser.add_argument('--dict_size', type=int, default=20)
parser.add_argument('--bins', type=int, default=1024)
parser.add_argument('--train_seed', type=int, default=44)
parser.add_argument('--dict_type', type=str, default='double_exponential')
parser.add_argument('--dict_shape_params', type=dict, 
    default={'mean1': 3e-6,'std1': 1e-7,'mean2': 1e-7,'std2': 1e-9})

parser.add_argument('--noise_unit', type=str, default='snr')

'''Test simulation parameters'''
parser.add_argument('--test_seed', type=int, default=42)
parser.add_argument('--test_sample', type=int, default=3000)
parser.add_argument('--test_lambda_n', type=float, default=0.06)

'''Network train parameters'''
parser.add_argument('--LR', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)


"""Parameters of the model"""
parser.add_argument('--problem_type', type=str, default='classification')
parser.add_argument('--model_name', type=str, default='UNetpp')
parser.add_argument('--model_depth', type=int, default=4) # Number of Level in the CNN Model
parser.add_argument('--model_width', type=int, default=128) # Width of the Initial Layer, subsequent layers start from here
parser.add_argument('--num_channel', type=int, default=1) # Input channel
parser.add_argument('--kernel_size', type=int, default=3) # Size of the Kernels/Filter
parser.add_argument('--alpha', type=int, default=1) # Model Width Expansion Parameter, for MultiResUNet only
parser.add_argument('--feature_number', type=int, default=1024) # Number of Features to be Extracted
parser.add_argument('--output_nums', type=int, default=1025)

parser.add_argument('--is_transconv', type=bool, default=True) # True: Transposed Convolution, False: UpSampling
parser.add_argument('--D_S', type=int, default=0) # Turn on Deep Supervision
parser.add_argument('--A_E', type=int, default=0) # Turn on AutoEncoder Mode for Feature Extraction
parser.add_argument('--A_G', type=int, default=0) # Turn on for Guided Attention
parser.add_argument('--LSTM', type=int, default=0) # Turn on for LSTM, Implemented for UNet and MultiResUNet only
parser.add_argument('--MHA', type=int, default=0) # Turn on Multi-Head Attention

parser.add_argument('--sample_length', type=int, default=1024)

parser.add_argument('--Lambda', type=int, default=0.1)


parser.add_argument('--N', type=int, default=2000)
parser.add_argument('--input_size', type=int, default=1024)
parser.add_argument('--dim_attn', type=int, default=64)
parser.add_argument('--dim_val', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
args = parser.parse_args()
