#!/bin/bash

# 参数扫描训练脚本
# 用法: ./run_parameter_sweep.sh <config_file> <parameter_name> <value1> <value2> ... <valueN>

CONFIG_DIR="/home/shendi_gjh_cj/codes/3D_project/configs"
TEMP_CONFIG_DIR="/tmp/parameter_sweep_configs"

# 检查参数数量
if [ $# -lt 3 ]; then
    echo "用法: $0 <config_file> <parameter_name> <value1> <value2> ... <valueN>"
    echo "示例: $0 configs/E2-01.json lr1 1e-4 1e-3 1e-2"
    echo "示例: $0 configs/E2-01.json stage1_epoch_number 100 200 300"
    exit 1
fi

CONFIG_FILE="$1"
PARAM_NAME="$2"
shift 2
VALUES=("$@")

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 创建临时配置目录
mkdir -p "$TEMP_CONFIG_DIR"

echo "开始参数扫描训练..."
echo "配置文件: $CONFIG_FILE"
echo "参数名称: $PARAM_NAME"
echo "参数值: ${VALUES[@]}"
echo "----------------------------------------"

# 为每个参数值创建临时配置文件并训练
for i in "${!VALUES[@]}"; do
    VALUE="${VALUES[$i]}"
    
    # 生成临时配置文件名称
    BASE_NAME=$(basename "$CONFIG_FILE" .json)
    TEMP_CONFIG="$TEMP_CONFIG_DIR/${BASE_NAME}_${PARAM_NAME}_${VALUE}.json"
    
    echo "处理参数值 $VALUE (${i+1}/${#VALUES[@]})"
    
    # 使用Python脚本修改配置文件
    python3 -c "
import json
import sys
import re

# 读取原始配置文件
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# 修改指定参数
config['$PARAM_NAME'] = $VALUE

# 写入临时配置文件
with open('$TEMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=4)

print(f'已创建临时配置文件: $TEMP_CONFIG')
print(f'参数 {PARAM_NAME} 已设置为: {VALUE}')
"
    
    # 运行训练
    echo "开始训练..."
    python train.py "$TEMP_CONFIG"
    
    if [ $? -eq 0 ]; then
        echo "✓ 参数值 $VALUE 训练完成"
    else
        echo "✗ 参数值 $VALUE 训练失败"
    fi
    
    echo "----------------------------------------"
done

# 清理临时文件
echo "清理临时配置文件..."
rm -rf "$TEMP_CONFIG_DIR"

echo "所有参数扫描训练完成！" 