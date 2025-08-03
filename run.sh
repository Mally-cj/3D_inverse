#!/bin/bash

CONFIG_DIR="/home/shendi_gjh_cj/codes/3D_project/configs"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Config directory $CONFIG_DIR does not exist."
  exit 1
fi

for config_file in "$CONFIG_DIR"/E1*.json; do
  [ -f "$config_file" ] || { echo "No E1*.json files found in $CONFIG_DIR"; break; }
  echo "Running training with config: $config_file"
  python train.py "$config_file" || echo "Error occurred with config: $config_file"
done

echo "All configurations processed."