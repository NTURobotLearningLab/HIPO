task_name=upNdown
env_scale=1
seed=42
option_num=3
cuda_devices="0"
lr=0.001

sh ./shell/high_level_policy_hipo_template.sh ${task_name} ${env_scale} ${seed} ${option_num} ${cuda_devices} ${lr}

