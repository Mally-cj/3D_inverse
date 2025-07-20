import numpy as np
import os
from seisvis.data_config import DataCube
from seisvis.plot2d import Seis2DPlotter
from seisvis.plot_config import PlotConfig
import torch
import sys
sys.path.append('/Users/chenjie53/Documents/3D_inverse/3D_inverse/deep_learning_impedance_inversion_chl')
from seisvis import plot1d

##判断当前环境是cpu还是gpu
if torch.cuda.is_available():
    USE_FULL_DATA = True
else:
    USE_FULL_DATA = False

# 加载3D低频背景阻抗，真实阻抗，预测阻抗
back_imp = np.load('background_impedance.npy')
true_imp = np.load('true_impedance.npy')
pred_imp = np.load('prediction_impedance.npy')  # shape: (N, T, S)





if not USE_FULL_DATA:
    well_positions = [[10, 10], [20, 20], [30, 30], [40, 40]]  # 适合(50, 251)网格的井位
else:
    basex = 450
    basey = 212
    pos = [[594,295], [572,692], [591,996], [532,1053], [603,1212], [561,842], [504,846], [499,597]]
    well_positions = [[y-basey, x-basex] for [x, y] in pos]



def plot_inline_sections_with_wells(pred_imp, true_imp, well_positions, save_dir='results'):
    """
    构造DataCube并画每口井所在的inline剖面，剖面上自动镶嵌该井曲线
    """
    # 构造DataCube
    cube = DataCube()
    cube.add_property('Predicted', pred_imp)
    cube.add_property('True', true_imp)
    for i, (inline, xline) in enumerate(well_positions):
        # 取该井的纵向曲线（time方向），用true_imp
        if 0 <= inline < true_imp.shape[0] and 0 <= xline < true_imp.shape[2]:
            well_log = true_imp[inline, :, xline].reshape(-1, 1)
        else:
            well_log = np.zeros((true_imp.shape[1], 1))
        cube.add_well(f'Well-{i+1}', {'log': well_log, 'coord': (inline, xline)})

    # 配置
    config = PlotConfig()
    size = [0, pred_imp.shape[0]-1, 0, pred_imp.shape[2]-1, pred_imp.shape[1]-1, 0]  # [il_start, il_end, xl_start, xl_end, t_end, t_start]
    plotter2d = Seis2DPlotter(cube, size, config)

    show_pred = {'type': 'Predicted', 'cmap': 'AI', 'clip': 'robust', 'mask': False, 'bar': True}
    wells_type = {'type': [f'Well-{i+1}' for i in range(len(well_positions))], 'cmap': 'AI', 'clip': None, 'width': 4}

    os.makedirs(save_dir, exist_ok=True)

    # 画每口井所在的inline剖面，剖面上自动镶嵌该井曲线
    for i, (inline, xline) in enumerate(well_positions):
        plotter2d.plot_section(
            section_idx=inline,
            section_type='inline',
            show_properties_type=show_pred,
            show_wells_type=wells_type,
            # title=f'Predicted Impedance Inline {inline}',
            save_path=f'{save_dir}/seisvis_pred_inline{inline}_with_well{i+1}.png'
        )
    print('✅ 每口井所在inline剖面图已保存到results目录')

def plot_well_curves_seisvis(true_imp, pred_imp, well_positions, back_imp=None, save_dir='results'):
    """
    用seisvis画每口井的真实/预测/低频背景曲线对比
    """
    config1d = PlotConfig()
    config1d.label_fontsize = 14
    config1d.tick_labelsize = 12
    config1d.legend_fontsize = 10
    plotter1d = plot1d.Seis1DPlotter(config1d)
    os.makedirs(save_dir, exist_ok=True)
    for i, (inline, xline) in enumerate(well_positions):
        if 0 <= inline < true_imp.shape[0] and 0 <= xline < true_imp.shape[2]:
            true_curve = true_imp[inline, :, xline].flatten()
            pred_curve = pred_imp[inline, :, xline].flatten()
            if back_imp is not None:
                back_curve = back_imp[inline, :, xline].flatten()
                data_groups = [[true_curve, pred_curve, back_curve]]
                legends = ['True', 'Predicted', 'Background']
                line_styles = ['-', '--', ':']
            else:
                data_groups = [[true_curve, pred_curve]]
                legends = ['True', 'Predicted']
                line_styles = ['-', '--']
            titles = [f'Well-{i+1} (inline={inline}, xline={xline})']
            plotter1d.plot_groups(
                data_groups=data_groups,
                t_start=0,
                titles=titles,
                legends=legends,
                line_styles=line_styles,
                vis_type='v',
                figsize=(4, 8),
                save_path=f'{save_dir}/well_curve_{i+1}_inline{inline}_xline{xline}.png'
            )
    print('✅ 每口井的1D曲线对比图已保存到results目录')



plot_inline_sections_with_wells(pred_imp, true_imp, well_positions, save_dir='results')

# 新增：画每口井的1D曲线对比（含低频背景）
plot_well_curves_seisvis(true_imp, pred_imp, well_positions, back_imp=back_imp, save_dir='results')




def plot_multiple_inlines_group_by_wells_seisvis(back_imp, true_imp, pred_imp, well_positions, save_dir='results'):
    """
    使用seisvis库，按井所在inline分为4组，每组分别画低频背景、真实阻抗、预测阻抗，每个属性单独保存为1个png
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    from seisvis.data_config import DataCube
    from seisvis.plot2d import Seis2DPlotter
    from seisvis.plot_config import PlotConfig

    # 统计每口井的inline
    well_inlines = [inline for inline, xline in well_positions]
    unique_inlines = sorted(list(set(well_inlines)))
    selected_inlines = unique_inlines[:4]
    prop_dict = {
        'Background': back_imp,
        'True': true_imp,
        'Predicted': pred_imp
    }
    titles = {'Background': 'Background', 'True': 'True (Interp)', 'Predicted': 'Predicted'}

    for inline in selected_inlines:
        # 只添加该inline上的井
        wells_in_inline = [(k, (w_inline, w_xline)) for k, (w_inline, w_xline) in enumerate(well_positions) if w_inline == inline]
        for prop_name, prop_data in prop_dict.items():
            # 构造DataCube
            cube = DataCube()
            cube.add_property(prop_name, prop_data)
            for k, (w_inline, w_xline) in wells_in_inline:
                well_log = true_imp[w_inline, :, w_xline].reshape(-1, 1)
                cube.add_well(f'Well-{k+1}', {'log': well_log, 'coord': (w_inline, w_xline)})
            config = PlotConfig()
            size = [0, prop_data.shape[0]-1, 0, prop_data.shape[2]-1, prop_data.shape[1]-1, 0]
            plotter2d = Seis2DPlotter(cube, size, config)
            show_prop = {'type': prop_name, 'cmap': 'AI', 'clip': 'robust', 'mask': False, 'bar': True}
            wells_type = {'type': [f'Well-{k+1}' for k, _ in wells_in_inline], 'cmap': 'AI', 'clip': None, 'width': 4}
            save_path = os.path.join(save_dir, f'inline_{inline}_{prop_name.lower()}_group_by_wells_seisvis.png')
            plotter2d.plot_section(
                section_idx=inline,
                section_type='inline',
                show_properties_type=show_prop,
                show_wells_type=wells_type,
                save_path=save_path
            )
    print(f'✅ 每个inline分组的剖面图（Background/True/Predicted）已分别保存到{save_dir}目录（seisvis版）')

# 调用新函数，按井inline分组画4组（每组3列：背景/真实/预测），每组单独保存，使用seisvis
plot_multiple_inlines_group_by_wells_seisvis(
    back_imp, true_imp, pred_imp, well_positions,
    save_dir='results'
)

def plot_grouped_inlines_matplotlib(back_imp, true_imp, pred_imp, well_positions, inline, save_path='results/grouped_inline.png'):
    """
    用最简单的matplotlib方式，在一个figure里画同一个inline的Background/True/Predicted剖面，并叠加井曲线，并加上颜色条。
    颜色条高度自适应和图片等高。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prop_dict = {
        'Background': back_imp,
        'True': true_imp,
        'Predicted': pred_imp
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    t_dim = back_imp.shape[1]
    x_dim = back_imp.shape[2]
    ims = []
    vmin = min([np.nanmin(prop[inline, :, :]) for prop in prop_dict.values()])
    vmax = max([np.nanmax(prop[inline, :, :]) for prop in prop_dict.values()])
    for j, (prop_name, prop_data) in enumerate(prop_dict.items()):
        section = prop_data[inline, :, :].T  # shape: (xline, t)
        im = axes[j].imshow(
            section,
            aspect='auto',
            cmap='seismic',
            origin='upper',
            extent=[0, t_dim-1, 0, x_dim-1],
            vmin=vmin,
            vmax=vmax
        )
        ims.append(im)
        axes[j].set_title(f'{prop_name} (inline={inline})')
        axes[j].set_xlabel('Time')
        if j == 0:
            axes[j].set_ylabel('Xline')
        # 叠加井曲线
        for k, (w_inline, w_xline) in enumerate(well_positions):
            if w_inline == inline:
                # 井曲线在该inline上，画在对应xline位置
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
plot_grouped_inlines_matplotlib(
    back_imp, true_imp, pred_imp, well_positions,
    inline=10,
    save_path='results/grouped_inline_10.png'
)
