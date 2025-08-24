import torch
import sys
sys.path.append("/home/shendi_gjh_cj/codes/3D_project/codes")
sys.path.append("/home/shendi_gjh_cj/codes/3D_project")

from wi_inv_model import UNet, DIPLoss
from wi_inv_socket import wi_inv_socket


from wi_inv_test import WIInv




predict = WIInv()