#!/bin/bash

# 直接在这里修改配置文件和参数
CONFIG_FILE="/home/shendi_gjh_cj/codes/3D_project/configs/E4.json"
# PARAM_NAME="tv_coeff"
# VALUES=(0.1 1 10 20 30 40 50)

PARAM_NAME="lr2"
VALUES=(1e-4 5e-5 2e-5 5e-4 1e-5)

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file $CONFIG_FILE does not exist."
  exit 1
fi

for val in "${VALUES[@]}"; do
  echo "Running training with $PARAM_NAME=$val"
  python3 -c "
import json, subprocess, os
config = json.load(open('$CONFIG_FILE'))
config['$PARAM_NAME'] = $val
temp_file = f'/tmp/E04_${PARAM_NAME}_${val}.json'
json.dump(config, open(temp_file, 'w'))
subprocess.run(['python', 'train.py', temp_file])
os.remove(temp_file)
" || echo "Error occurred with $PARAM_NAME=$val"
done

echo "All parameter values processed." 