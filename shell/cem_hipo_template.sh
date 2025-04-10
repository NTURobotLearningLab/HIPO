task_name=${1} # infinite_harvester, upNdown, infinite_doorkey, seesaw, farmer
env_scale=${2} # 2: infinite_harvester, seesaw, 1: else
height=$((8*env_scale)) 
width=$((8*env_scale))
seed=${3}

option_num=${4}
diversity_num=${5}
outdir="result/${task_name}_cem_option${option_num}_seed${seed}/vector${diversity_num}"
configfile="config_cem.py"
vector_file_name="final_vectors.pkl"

echo $outdir
echo $configfile

num_demo_per_program=${6}
cuda_devices=${7}
random_seq_min_number=${8}
max_number_of_epochs=${9}

if [ ${option_num} -eq 1 ]
then
  echo "Option 1"
  option_dir=None
else
  echo "Option > 1"
  root_path="result/${task_name}_cem_option$((option_num-1))_seed${seed}"
  temp_file="option_path_${task_name}_cem_option${option_num}_seed${seed}.txt"
  python get_vector_path.py ${root_path} ${vector_file_name} > ${temp_file}
  read option_dir < ${temp_file}
  echo $option_dir
  rm -rf ${temp_file}
fi

if [ ${diversity_num} -eq 1 ]
then
  echo "vector 1"
  python random_seq.py $((option_num-1)) ${random_seq_min_number} ${seed} > random_seq_${task_name}_seed${seed}_option${option_num}.txt
  diversity_dir=None

else
  echo "vector > 1"
  root_path="result/${task_name}_cem_option${option_num}_seed${seed}/vector$((diversity_num-1))"
  temp_file="diversity_vector_path_${task_name}_cem_option${option_num}_seed${seed}_vector${diversity_num}.txt"
  
  python get_vector_path.py ${root_path} ${vector_file_name} > ${temp_file}
  read diversity_dir < ${temp_file}
  echo $diversity_dir
  rm -rf ${temp_file}
fi

population_size=64
sigma=0.5
elitism_rate=0.05
use_exp_sig_decay=True
init_type=normal

echo option${option_num} vector${vector} === $population_size $sigma $elitism_rate $use_exp_sig_decay $init_type ===

CUDA_VISIBLE_DEVICES=$cuda_devices python main.py  --configfile $configfile \
--net.saved_params_path leaps/weights/LEAPS/best_valid_params.ptp \
--seed $seed \
--algorithm CEM \
--outdir $outdir \
--env_task $task_name \
--CEM.max_number_of_epochs $max_number_of_epochs \
--CEM.population_size $population_size \
--CEM.sigma $sigma \
--CEM.elitism_rate $elitism_rate \
--CEM.init_type $init_type \
--CEM.use_exp_sig_decay $use_exp_sig_decay \
--CEM.early_stop_std 0.001 \
--CEM.vertical_similarity True \
--CEM.random_seq random_seq_${task_name}_seed${seed}_option${option_num}.txt \
--num_demo_per_program $num_demo_per_program \
--rl.envs.executable.num_demo_per_program $num_demo_per_program \
--height $height \
--width $width \
--CEM.diversity_vector $diversity_dir \
--CEM.compatibility_vector $option_dir




