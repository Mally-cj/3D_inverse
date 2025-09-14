from data_tools import *


import os
import numpy as np
from icecream import ic

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

folder="/home/shendi_gjh_cj/codes/3D_project/logs/E11_1_1_0.5_0_4"

# def show(folder):
idx=10
name = folder.split("E", 1)[1].split("/")[0]
pic_folder=os.path.join(folder,f"{name}_pic_folder")
os.makedirs(pic_folder,exist_ok=True)

imp_3d=np.load(os.path.join(folder,"prediction_impedance.npy"))

seis_3d=np.load(os.path.join(folder,"seismic_record.npy"))
true_imp_3d=np.load(os.path.join(folder,"true_impedance.npy"))
low_imp_3d=np.load(os.path.join(folder,"implow_impedance.npy"))
re_seismic_3d=np.load(os.path.join(folder,"re_seismic_record.npy"))


fan_min= 7.9577527
fan_max=9.507863
true_imp_3d=np.exp(true_imp_3d*(fan_max-fan_min)+fan_min)
imp_3d=np.exp(imp_3d*(fan_max-fan_min)+fan_min)
low_imp_3d=np.exp(low_imp_3d*(fan_max-fan_min)+fan_min)

single_imshow(true_imp_3d[:,:,idx],cmap=plt.cm.jet,title=f'true_imp_{idx}')
single_imshow(low_imp_3d[:,:,idx],cmap=plt.cm.jet,title=f'low_impedance_{idx}')

for idx in range(180,190,10):
#     single_imshow(seis_3d[:,idx,:],vmin=-0.22,vmax=0.22,cmap=plt.cm.seismic,title=f'seismic_xline{idx}',
# save=True,save_dir=pic_folder)
#     vmin=imp_3d[idx,:,:].min()
#     vmax=imp_3d[idx,:,:].max()
#     single_imshow(imp_3d[:,:,idx],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'E{name}_impedance_{idx}_vmin={vmin:.3f}_vmax={vmax:.3f}',
# save=True,save_dir=pic_folder)
    
    imp=imp_3d[400:500,10:300,idx]
    vmin=imp.min()
    vmax=imp.max()
#     single_imshow(imp,cmap=plt.cm.jet,title=f'E{name}_imp_time{idx}_vmin={vmin:.3f}_vmax={vmax:.3f}',
# save=True,save_dir=pic_folder)
#     single_imshow(true_imp_3d[:,idx,:],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'true_impedance_xline{idx}',
# save=True,save_dir=pic_folder)
#     single_imshow(low_imp_3d[:,idx,:],vmin=0.4,vmax=1.0,cmap=plt.cm.jet,title=f'low_impedance_xline{idx}',
# save=True,save_dir=pic_folder)
# ic(seis_3d.shape)    

# plt.plot(imp_3d[:,0,idx].flatten(),label="pred")
# plt.plot(true_imp_3d[:,0,idx].flatten(),label="true")
# plt.plot(low_imp_3d[:,0,idx].flatten(),label="low")
# plt.legend()
# plt.show()
# imp_3d=imp_3d[300:450]
# true_imp_3d=true_imp_3d[300:450]
# seis_3d=seis_3d[300:450]
# pdb.set_trace()
# plt.close()
# plt.plot(imp_3d[:,0,idx].flatten())
# plt.plot(true_imp_3d[:,0,idx].flatten())


plot_sections_with_wells_single(
    pred_imp=imp_3d, 
    true_imp=true_imp_3d, 
    re_seismic=seis_3d, 
    well_pos=None, 
    section_type='xline', 
    save_dir=pic_folder,
    show_well=True)
plot_well_curves_seisvis(true_imp_3d, imp_3d, well_pos=None, back_imp=None, save_dir=pic_folder)

# plt.plot(imp_3d[:,634,90:100:5])
# # plt.plot(imp_3d[:,634,109].flatten())
# plt.legend()
# plt.show()

# folder="/home/shendi_gjh_cj/codes/3D_project/logs/E11_5_5_20_0_4"
# show("/home/shendi_gjh_cj/codes/3D_project/logs/E11_1_1_0.5_0_4")
# show("/home/shendi_gjh_cj/codes/3D_project/logs/E11_5_5_20_0_4")


mask_3d=np.load("/home/shendi_gjh_cj/codes/3D_project/mask_grid.npy")
# single_imshow(mask_3d,title="mask_3d")



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






