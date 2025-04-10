task_name=${1} # infinite_harvester, upNdown, infinite_doorkey, seesaw, farmer
env_scale=${2} # 2: infinite_harvester, seesaw, 1: else
height=$((8*env_scale)) 
width=$((8*env_scale))
seed=${3}

option_num=${4}
outdir="result/${task_name}_high_level_policy_option${option_num}_seed${seed}"
configfile="config_high_level_policy.py"
vector_file_name="final_vectors.pkl"

echo $outdir
echo $configfile

cuda_devices=${5}
lr=${6}

root_path="result/${task_name}_cem_option${option_num}_seed${seed}"
temp_file="option_path_${task_name}_high_level_policy_option${option_num}_seed${seed}.txt"
python get_vector_path.py ${root_path} ${vector_file_name} > ${temp_file}
read option_dir < ${temp_file}
echo $option_dir
rm -rf ${temp_file}

CUDA_VISIBLE_DEVICES=$cuda_devices python main.py  --configfile $configfile \
--env_task $task_name \
--net.saved_params_path leaps/weights/LEAPS/best_valid_params.ptp \
--seed $seed \
--algorithm HighLevelPolicy \
--outdir $outdir \
--option_num $option_num \
--CEM.compatibility_vector $option_dir \
--PPO.lr $lr \
--height $height \
--width $width
