#!/bin/bash
# 简化版：使用固定配置文件，修改指定参数的不同值并训练

# 直接指定配置文件路径（在这里写入你的JSON文件路径）
CONFIG="/home/shendi_gjh_cj/codes/3D_project/configs/E3-01.json"

[ ! -f "$CONFIG" ] && { echo "Config $CONFIG not found"; exit 1; }
command -v jq &>/dev/null || { echo "Need jq installed"; exit 1; }

# 要测试的参数和值（修改这里即可）
PARAM="tv_coeff"
VALUES=(10 20 30 40 50)

# 临时文件前缀
PREFIX=$(mktemp)

# 循环测试每个值
for val in "${VALUES[@]}"; do
  TEMP_FILE="${PREFIX}_${PARAM}_${val}.json"
  jq ".$PARAM = $val" "$CONFIG" > "$TEMP_FILE" && {
    echo "Training with $PARAM=$val..."
    python train.py "$TEMP_FILE" || echo "Error with $val"
  }
done

# 清理临时文件
rm -f "${PREFIX}_"*.json

echo "Done"
