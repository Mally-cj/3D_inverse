from data_tools import *


import os
import numpy as np


# folder="logs/E9-4/test/"
# impedance_3d=np.load(os.path.join(folder,"test_epoch=119",'prediction_impedance.npy'))

# idx=350;
# single_imshow(impedance_3d[idx,:,:],vmin=0.4,vmax=1,cmap=plt.cm.jet,title=f'impedance:{idx}')

# seismic_3d=np.load(os.path.join(folder,"test_epoch=0",'seismic_record.npy'))

# # delta_seismic=seismic_3d[:,:,:-2]-seismic_3d[:,:,1:-1]
# single_imshow(seismic_3d[idx,:],vmin=-0.22,vmax=0.22,cmap=plt.cm.seismic,title=f'seismic:{idx}')



# imp=np.load("/home/shendi_gjh_cj/codes/3D_project/logs/tmp/prediction_impedance.npy")
# single_imshow(imp[:,:,0],vmin=0.4,vmax=1,cmap=plt.cm.jet,title=f'impedance')



import os
import numpy as np
import matplotlib.pyplot as plt
from visual_results import plot_sections_with_wells,plot_well_curves_seisvis,plot_sections_with_wells_single

from data_tools import *


# # def show(folder):
# idx=141
# folder="/home/shendi_gjh_cj/codes/3D_project/logs/E11-4"
# name = folder.split("E", 1)[1].split("/")[0]
# imp_3d=np.load(os.path.join(folder,"prediction_impedance.npy"))
# single_imshow(imp_3d[:,:,idx],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'E{name}_impedance_{idx}')
# plt.plot(imp_3d[:,0,idx].flatten())



# plot_sections_with_wells(imp_3d, true_imp_3d, low_imp_3d, seis_3d, well_pos=None, 
#             section_type='inline', save_dir=folder)


def show(folder):
    idx=141
    name = folder.split("E", 1)[1].split("/")[0]
    imp_3d=np.load(os.path.join(folder,"prediction_impedance.npy"))
    single_imshow(imp_3d[:,:,idx],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'E{name}_impedance_{idx}')
 
    seis_3d=np.load(os.path.join(folder,"seismic_record.npy"))
    true_imp_3d=np.load(os.path.join(folder,"true_impedance.npy"))
    low_imp_3d=np.load(os.path.join(folder,"implow_impedance.npy"))
    single_imshow(seis_3d[:,:,idx],vmin=-0.22,vmax=0.22,cmap=plt.cm.seismic,title=f'seismic_{idx}')
    single_imshow(true_imp_3d[:,:,idx],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'true_impedance_{idx}')
    single_imshow(low_imp_3d[:,:,idx],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'low_impedance_{idx}')

    plot_sections_with_wells(imp_3d, true_imp_3d, low_imp_3d, seis_3d, well_pos=None, 
            section_type='inline', save_dir=folder)
    plot_well_curves_seisvis(true_imp_3d, imp_3d, well_pos=None, back_imp=None, save_dir=folder)

show("/home/shendi_gjh_cj/codes/3D_project/logs/E11-4")
# show("/home/shendi_gjh_cj/codes/3D_project/logs/E11-2")
# show("/home/shendi_gjh_cj/codes/3D_project/logs/E9-5/test/test_epoch=119")

    # seis_3d=np.load(os.path.join(folder,"seismic_record.npy"))
    # true_imp_3d=np.load(os.path.join(folder,"true_impedance.npy"))
    # low_imp_3d=np.load(os.path.join(folder,"implow_impedance.npy"))
    # single_imshow(seis_3d[:,:,idx],vmin=-0.22,vmax=0.22,cmap=plt.cm.seismic,title=f'seismic_{idx}')
    # single_imshow(true_imp_3d[:,:,idx],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'true_impedance_{idx}')
    # single_imshow(low_imp_3d[:,:,idx],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'low_impedance_{idx}')

    # plot_sections_with_wells(imp_3d, true_imp_3d, low_imp_3d, seis_3d, well_pos=None, 
    #         section_type='inline', save_dir=folder)

# show("/home/shendi_gjh_cj/codes/3D_project/logs/E11-1")
# show("/home/shendi_gjh_cj/codes/3D_project/logs/E11-2")
# show("/home/shendi_gjh_cj/codes/3D_project/logs/E11-3")
# show("/home/shendi_gjh_cj/codes/3D_project/logs/E9-5/test/test_epoch=119")






