#!/bin/bash
# This script displays GPU information in a more fine grained way.
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | while IFS=, read -r gpu_uuid pid process_name used_memory; do
  gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total,gpu_uuid --format=csv,noheader | grep "$gpu_uuid")
  username=$(ps -p $pid -o user --no-headers)
  echo "User: $username, PID: $pid, Process: $process_name, Used Memory: $used_memory MB, GPU-Info: $gpu_info"
done
