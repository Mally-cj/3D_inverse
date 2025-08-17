import numpy as np
import os
from seisvis.data_config import DataCube
from seisvis.plot2d import Seis2DPlotter
from seisvis.plot_config import PlotConfig
import torch
import sys
from seisvis import plot1d
import pdb
import matplotlib.pyplot as plt
import data_tools as tools
import pdb






##判断当前环境是cpu还是gpu
if torch.cuda.is_available():
    USE_FULL_DATA = True
else:
    USE_FULL_DATA = True

# 加载3D低频背景阻抗，真实阻抗，预测阻抗，地震数据
back_imp = np.load('logs/results/background_impedance.npy')
true_imp = np.load('logs/results/true_impedance.npy')
pred_imp = np.load('logs/results/prediction_impedance.npy')  # shape: (time, CMP, inline)
seismic = np.load('logs/results/seismic_record.npy')  # shape: (time, CMP, inline)

# 井位坐标定义
# 根据readme.txt，井位基于Line(450-700)和CMP(212-1400)范围
if not USE_FULL_DATA:
    well_positions = [[10, 10], [20, 20], [30, 30], [40, 40]]  # 适合(50, 251)网格的井位
else:
    base_line = 450   # Line起始值
    base_cmp = 212    # CMP起始值
    # 原始井位坐标 (Line, CMP)
    pos = [[594,295], [572,692], [591,996], [532,1053], [603,1212], [561,842], [504,846], [499,597]]
    # 转换为相对坐标 (inline_idx, xline_idx)
    well_positions = [[line-base_line, cmp-base_cmp] for [line, cmp] in pos]
    print("井位相对坐标 (inline_idx, xline_idx):", well_positions)


def plot_well_curves_seisvis(true_imp, pred_imp, well_pos, back_imp=None, save_dir='logs/results'):
    """
    用seisvis画每口井的真实/预测/低频背景曲线对比
    
    Args:
        true_imp: 真实阻抗数据，shape (time, CMP, inline)
        pred_imp: 预测阻抗数据，shape (time, CMP, inline)
        well_positions: 井位列表，每个元素为 (inline_idx, xline_idx)
        back_imp: 低频背景阻抗数据，shape (time, CMP, inline)
        save_dir: 保存目录
    """
    if well_pos is None:
        well_pos=well_positions
    config1d = PlotConfig()
    config1d.label_fontsize = 14
    config1d.tick_labelsize = 12
    config1d.legend_fontsize = 10
    plotter1d = plot1d.Seis1DPlotter(config1d)
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (inline_idx, xline_idx) in enumerate(well_pos):
        if 0 <= inline_idx < true_imp.shape[2] and 0 <= xline_idx < true_imp.shape[1]:
            # 提取井曲线：固定inline和xline位置，取所有时间点的值
            true_curve = true_imp[:, xline_idx, inline_idx].flatten()
            pred_curve = pred_imp[:, xline_idx, inline_idx].flatten()
            
            if back_imp is not None:
                back_curve = back_imp[:, xline_idx, inline_idx].flatten()
                data_groups = [[true_curve, pred_curve, back_curve]]
                legends = ['True', 'Predicted', 'Background']
                line_styles = ['-', '--', ':']
            else:
                data_groups = [[true_curve, pred_curve]]
                legends = ['True', 'Predicted']
                line_styles = ['-', '--']
            
            from scipy.stats import pearsonr

            ##计算true_curve和pred_curve的相关性
            corr, p_value = pearsonr(true_curve, pred_curve)
            titles = [f'Well-{i+1} (inline={inline_idx}, xline={xline_idx}, corr={corr:.2f}, p={p_value:.2f})']
            plotter1d.plot_groups(
                data_groups=data_groups,
                t_start=0,
                titles=titles,
                legends=legends,
                line_styles=line_styles,
                vis_type='v',
                figsize=(4, 8),
                save_path=f'{save_dir}/well_curve_{i+1}_inline{inline_idx}_xline{xline_idx}.png'
            )
    print('✅ 每口井的1D曲线对比图已保存到results目录')


# 新增：画每口井的1D曲线对比（含低频背景）
# plot_well_curves_seisvis(true_imp, pred_imp, well_positions, back_imp=back_imp, save_dir='results')

def plot_multiple_inlines_group_by_wells_seisvis(back_imp, true_imp, pred_imp, seismic, well_positions, save_dir='results'):
    """
    使用seisvis库，按井所在inline分为4组，每组分别画低频背景、真实阻抗、预测阻抗，每个属性单独保存为1个png
    
    Args:
        back_imp: 低频背景阻抗数据，shape (time, CMP, inline)
        true_imp: 真实阻抗数据，shape (time, CMP, inline)
        pred_imp: 预测阻抗数据，shape (time, CMP, inline)
        seismic: 地震数据，shape (time, CMP, inline)
        well_positions: 井位列表，每个元素为 (inline_idx, xline_idx)
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    from seisvis.data_config import DataCube
    from seisvis.plot2d import Seis2DPlotter
    from seisvis.plot_config import PlotConfig

    # 统计每口井的inline
    well_inlines = [inline_idx for inline_idx, xline_idx in well_positions]
    unique_inlines = sorted(list(set(well_inlines)))
    selected_inlines = unique_inlines[:4]
    prop_dict = {
        'Background': back_imp,
        'True': true_imp,
        'Predicted': pred_imp
    }
    titles = {'Background': 'Background', 'True': 'True (Interp)', 'Predicted': 'Predicted'}

    for inline_idx in selected_inlines:
        # 只添加该inline上的井
        wells_in_inline = [(k, (w_inline, w_xline)) for k, (w_inline, w_xline) in enumerate(well_positions) if w_inline == inline_idx]
        for prop_name, prop_data in prop_dict.items():
            # 构造DataCube
            cube = DataCube()
            cube.add_property(prop_name, prop_data)
            # 为当前inline上的每口井添加井曲线数据
            # for k, (w_inline, w_xline) in wells_in_inline:
            #     well_log = true_imp[:, w_xline, w_inline].reshape(-1, 1)
            #     cube.add_well(f'Well-{k+1}', {'log': well_log, 'coord': (w_inline, w_xline)})
            config = PlotConfig()
            size = [0, prop_data.shape[2]-1, 0, prop_data.shape[1]-1, prop_data.shape[0]-1, 0]
            plotter2d = Seis2DPlotter(cube, size, config)
            show_prop = {'type': prop_name, 'cmap': 'AI', 'clip': 'robust', 'mask': False, 'bar': True}
            wells_type = {'type': [f'Well-{k+1}' for k, _ in wells_in_inline], 'cmap': 'AI', 'clip': None, 'width': 4}
            save_path = os.path.join(save_dir, f'inline_{inline_idx}_{prop_name.lower()}_group_by_wells_seisvis.png')
            plotter2d.plot_section(
                section_idx=inline_idx,
                section_type='inline',
                show_properties_type=show_prop,
                show_wells_type=wells_type,
                save_path=save_path
            )
    print(f'✅ 每个inline分组的剖面图（Background/True/Predicted）已分别保存到{save_dir}目录（seisvis版）')

# # 调用新函数，按井inline分组画4组（每组4列：背景/真实/预测/地震），每组单独保存，使用seisvis
# plot_multiple_inlines_group_by_wells_seisvis(
#     back_imp, true_imp, pred_imp, seismic, well_positions,
#     save_dir='results')

def plot_grouped_inlines_matplotlib(back_imp, true_imp, pred_imp, well_positions, inline_idx, save_path='results/grouped_inline.png'):
    """
    用最简单的matplotlib方式，在一个figure里画同一个inline的Background/True/Predicted剖面，并叠加井曲线，并加上颜色条。
    颜色条高度自适应和图片等高。
    
    Args:
        back_imp: 低频背景阻抗数据，shape (time, CMP, inline)
        true_imp: 真实阻抗数据，shape (time, CMP, inline)
        pred_imp: 预测阻抗数据，shape (time, CMP, inline)
        well_positions: 井位列表，每个元素为 (inline_idx, xline_idx)
        inline_idx: 要绘制的inline索引
        save_path: 保存路径
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prop_dict = {
        'Background': back_imp,
        'True': true_imp,
        'Predicted': pred_imp
    }
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    t_dim = back_imp.shape[0]  # 时间维度
    x_dim = back_imp.shape[1]  # CMP维度
    ims = []
    vmin = min([np.nanmin(prop[:, :, inline_idx]) for prop in prop_dict.values()])
    vmax = max([np.nanmax(prop[:, :, inline_idx]) for prop in prop_dict.values()])
    
    for j, (prop_name, prop_data) in enumerate(prop_dict.items()):
        # 提取inline剖面：固定inline，显示所有CMP和时间
        section = prop_data[:, :, inline_idx].T  # shape: (CMP, time)
        im = axes[j].imshow(
            section,
            aspect='auto',
            cmap=plt.cm.jet,
            origin='upper',
            extent=[0, t_dim-1, 0, x_dim-1],
            vmin=vmin,
            vmax=vmax
        )
        ims.append(im)
        axes[j].set_title(f'{prop_name} (inline={inline_idx})')
        axes[j].set_xlabel('Time')
        if j == 0:
            axes[j].set_ylabel('CMP')
        # 叠加井曲线
        for k, (w_inline, w_xline) in enumerate(well_positions):
            if w_inline == inline_idx:
                # 井曲线在该inline上，画在对应CMP位置
                axes[j].plot(np.arange(t_dim), np.full(t_dim, w_xline), 'k--', lw=2, alpha=0.7)
        # 添加颜色条（等高）
        divider = make_axes_locatable(axes[j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Impedance', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ 已保存分组inline剖面图到 {save_path}')

## 调用plot_grouped_inlines_matplotlib
# 示例调用：画inline=10的分组剖面图
# plot_grouped_inlines_matplotlib(
#     back_imp, true_imp, pred_imp, well_positions,
#     inline_idx=10,
#     save_path='logs/results/grouped_inline_10.png')



def plot_sections_with_wells_single(pred_imp, true_imp, well_pos=None,section_type='inline', save_dir='results',epoch=0,show_well=False):
    """
    构造DataCube并画指定方向的剖面，剖面上自动镶嵌井曲线
    
    Args:
        pred_imp: 预测阻抗数据，shape (time, CMP, inline)
        true_imp: 由测井数据插值得到的阻抗数据，shape (time, CMP, inline)
        well_positions: 井位列表，每个元素为 (inline_idx, xline_idx)
        section_type: 剖面类型，'inline' 或 'xline'
        save_dir: 保存目录
    """
    # 构造DataCube
    cube = DataCube()
    cube.add_property('Predicted', pred_imp.transpose(2, 1, 0))  # 转换为 (inline, xline, time) 形状
    
    if well_pos is None:
        well_pos=well_positions

    # 添加井曲线
    for i, (inline_idx, xline_idx) in enumerate(well_pos):
        if 0 <= inline_idx < true_imp.shape[2] and 0 <= xline_idx < true_imp.shape[1]:
            well_log = true_imp[:, xline_idx, inline_idx].reshape(-1, 1)
            cube.add_well(f'Well-{i+1}', {'log': well_log, 'coord': (inline_idx, xline_idx)})

    # 配置seisvis参数
    config = PlotConfig()
    size = [0, pred_imp.shape[2]-1, 0, pred_imp.shape[1]-1, pred_imp.shape[0]-1, 0]
    plotter2d = Seis2DPlotter(cube, size, config)

    ##求pred，true，back的vmin，vmax
    vmin = min(np.nanmin(pred_imp), np.nanmin(true_imp), np.nanmin(back_imp))
    vmax = max(np.nanmax(pred_imp), np.nanmax(true_imp), np.nanmax(back_imp))
    
    # 显示配置
    show_pred = {'type': 'Predicted', 'cmap': 'AI', 'clip':(vmin,vmax), 'mask': False, 'bar': True}
    wells_type = {'type': [f'Well-{i+1}' for i in range(len(well_pos))], 'cmap': 'AI', 'clip': (vmin,vmax), 'width': 4}

    if show_well is False:
        wells_type = None
    
    os.makedirs(save_dir, exist_ok=True)

    # 获取需要绘制的剖面位置
    if section_type == 'inline':
        section_positions = list(set([inline_idx for inline_idx, _ in well_pos]))
        x_label = 'CMP'
    else:  # xline
        section_positions = list(set([xline_idx for _, xline_idx in well_pos]))
        x_label = 'Inline'

    # 绘制每个剖面
    for i,section_idx in enumerate(section_positions):
        # 绘制预测和真实阻抗剖面
        for imp_type, show_config in [('pred', show_pred)]:
            plotter2d.plot_section(
                section_idx=section_idx,
                section_type=section_type,
                show_properties_type=show_config,
                show_wells_type=wells_type,
                save_path=f'{save_dir}/well{i}_{imp_type}_{section_type}{section_idx}.png',
                title_define=f'{imp_type.title()} Impedance ({section_type.title()} {section_idx},epoch={epoch})'
            )
        # if i> 5: break
        
    
    print(f'✅ {section_type}方向剖面图已保存到{save_dir}目录')



def plot_sections_with_wells(pred_imp, true_imp, back_imp,seismic, well_pos=None,section_type='inline', save_dir='results',show_well=False):
    """
    构造DataCube并画指定方向的剖面，剖面上自动镶嵌井曲线
    
    Args:
        pred_imp: 预测阻抗数据，shape (time, CMP, inline)
        true_imp: 由测井数据插值得到的阻抗数据，shape (time, CMP, inline)
        well_positions: 井位列表，每个元素为 (inline_idx, xline_idx)
        section_type: 剖面类型，'inline' 或 'xline'
        save_dir: 保存目录
    """
    # 构造DataCube
    cube = DataCube()
    cube.add_property('Predicted', pred_imp.transpose(2, 1, 0))  # 转换为 (inline, xline, time) 形状
    cube.add_property('True', true_imp.transpose(2, 1, 0))
    cube.add_property('Seismic', seismic.transpose(2, 1, 0))
    cube.add_property('Background', back_imp.transpose(2, 1, 0))
    
    if well_pos is None:
        well_pos=well_positions

    # 添加井曲线
    for i, (inline_idx, xline_idx) in enumerate(well_pos):
        if 0 <= inline_idx < true_imp.shape[2] and 0 <= xline_idx < true_imp.shape[1]:
            well_log = true_imp[:, xline_idx, inline_idx].reshape(-1, 1)
            cube.add_well(f'Well-{i+1}', {'log': well_log, 'coord': (inline_idx, xline_idx)})

    # 配置seisvis参数
    config = PlotConfig()
    size = [0, pred_imp.shape[2]-1, 0, pred_imp.shape[1]-1, pred_imp.shape[0]-1, 0]
    plotter2d = Seis2DPlotter(cube, size, config)

    ##求pred，true，back的vmin，vmax
    vmin = min(np.nanmin(pred_imp), np.nanmin(true_imp), np.nanmin(back_imp))
    vmax = max(np.nanmax(pred_imp), np.nanmax(true_imp), np.nanmax(back_imp))
    
    # 显示配置
    show_pred = {'type': 'Predicted', 'cmap': 'AI', 'clip':(vmin,vmax), 'mask': False, 'bar': True}
    show_true = {'type': 'True', 'cmap': 'AI', 'clip': (vmin,vmax), 'mask': False, 'bar': True}
    show_back = {'type': 'Background', 'cmap': 'AI', 'clip': (vmin,vmax), 'mask': False, 'bar': True}
    wells_type = {'type': [f'Well-{i+1}' for i in range(len(well_pos))], 'cmap': 'AI', 'clip': (vmin,vmax), 'width': 4}
    show_seismic = {'type': 'Seismic', 'cmap': 'Grey_scales', 'clip': 'robust', 'mask': False, 'bar': True}

    if show_well is False:
        wells_type = None
    
    os.makedirs(save_dir, exist_ok=True)

    # 获取需要绘制的剖面位置
    if section_type == 'inline':
        section_positions = list(set([inline_idx for inline_idx, _ in well_pos]))
        x_label = 'CMP'
    else:  # xline
        section_positions = list(set([xline_idx for _, xline_idx in well_pos]))
        x_label = 'Inline'

    # 绘制每个剖面
    for i,section_idx in enumerate(section_positions):
        # 绘制预测和真实阻抗剖面
        for imp_type, show_config in [('pred', show_pred), ('true', show_true), ('back', show_back), ('seismic', show_seismic)]:
            plotter2d.plot_section(
                section_idx=section_idx,
                section_type=section_type,
                show_properties_type=show_config,
                show_wells_type=wells_type,
                save_path=f'{save_dir}/well{i}_{imp_type}_{section_type}{section_idx}.png',
                title_define=f'{imp_type.title()} Impedance ({section_type.title()} {section_idx})'
            )
        # if i> 2: break
        
    
    print(f'✅ {section_type}方向剖面图已保存到{save_dir}目录')

# 调用示例
# plot_sections_with_wells(pred_imp, true_imp,back_imp, seismic, well_positions, section_type='inline', save_dir='logs/results')
# plot_sections_with_wells(pred_imp, true_imp,back_imp,seismic,  well_positions, section_type='xline', save_dir='logs/results')





from utils import save_stage2_loss_data, save_complete_training_loss
from data_tools import ProcessRunner

class Visual_runner(ProcessRunner):
    def __init__(self):
        super().__init__()

    def _init_worker(self):
        pass

    def _run(self, save_dir, stage2_total_loss, stage2_sup_loss, stage2_unsup_loss, stage2_tv_loss,total_lossF):
        save_stage2_loss_data(save_dir, stage2_total_loss, stage2_sup_loss, 
                                stage2_unsup_loss, stage2_tv_loss)    
    #     # 保存完整训练过程loss对比图
        save_complete_training_loss(save_dir, total_lossF, stage2_total_loss, 
                                    stage2_sup_loss, stage2_unsup_loss, stage2_tv_loss, 
                                    )
        pass