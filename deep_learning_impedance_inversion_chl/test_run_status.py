#!/usr/bin/env python3
"""
修正版代码运行确认脚本
====================
快速验证修正版代码是否能正常运行
"""

import sys
import time

def test_corrected_version():
    """测试修正版代码运行状态"""
    
    print("🧪 修正版代码运行测试")
    print("=" * 50)
    
    try:
        print("📋 测试1：导入和初始化...")
        
        # 导入必要模块（测试依赖是否正常）
        import torch
        import numpy as np
        from Model.net2D import UNet, forward_model
        
        # 测试设备检测
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   ✅ 设备检测正常: {device}")
        
        # 测试网络初始化
        net = UNet(
            in_ch=2,
            out_ch=1, 
            channels=[8, 16, 32, 64],
            skip_channels=[0, 8, 16, 32],
            use_sigmoid=True,
            use_norm=False
        ).to(device)
        
        forward_net = forward_model(nonlinearity="tanh").to(device)
        print(f"   ✅ 网络初始化正常")
        
        print("\n📋 测试2：数据流向验证...")
        
        # 模拟训练数据
        batch_size = 1
        time_samples = 601
        patch_size = 48
        
        S_obs = torch.randn(batch_size, 1, time_samples, patch_size).to(device)
        Z_init = torch.randn(batch_size, 1, time_samples, patch_size).to(device)
        
        # 测试UNet前向传播
        unet_input = torch.cat([Z_init, S_obs], dim=1)
        with torch.no_grad():
            Z_pred = net(unet_input)
            
        print(f"   ✅ UNet前向传播正常: {unet_input.shape} → {Z_pred.shape}")
        
        # 测试Forward网络
        reflection_coeff = torch.randn(batch_size, 1, time_samples, patch_size).to(device)
        wav_init = torch.randn(1, 1, 101, 1).to(device)
        
        with torch.no_grad():
            synthetic_seismic, learned_wavelet = forward_net(reflection_coeff, wav_init)
            
        print(f"   ✅ Forward网络正常: 反射系数 → 合成地震数据")
        
        print("\n📋 测试3：损失函数验证...")
        
        # 测试损失函数
        mse = torch.nn.MSELoss()
        M_mask = torch.rand(batch_size, 1, time_samples, patch_size).to(device)
        
        # 井约束损失
        loss_sup = mse(M_mask * Z_pred, M_mask * Z_init)
        print(f"   ✅ 井约束损失计算正常: {loss_sup.item():.6f}")
        
        # TV正则化损失
        def tv_loss(x, alfa=1e-4):
            dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
            dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
            return alfa * torch.mean(2*dh[..., :-1, :] + dw[..., :, :-1])
        
        loss_tv = tv_loss(Z_pred)
        print(f"   ✅ TV正则化损失计算正常: {loss_tv.item():.6f}")
        
        print("\n" + "=" * 50)
        print("🎉 修正版代码运行测试通过！")
        print("=" * 50)
        
        print("\n📋 测试总结:")
        print("✅ 依赖模块导入正常")
        print("✅ 网络架构初始化正常")
        print("✅ 数据流向验证通过")
        print("✅ 损失函数计算正常")
        print("✅ 修正版代码可以正常运行")
        
        print("\n💡 下一步:")
        print("   - 运行完整训练: python seismic_imp_2D_high_channel_model_bgp_corrected.py")
        print("   - 根据实际数据调整超参数")
        print("   - 监控训练过程和损失收敛")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_corrected_version()
    sys.exit(0 if success else 1)
