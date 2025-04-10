task_name=farmer
env_scale=1
seed=42
num_demo_per_program=1
cuda_devices="0"
max_number_of_epochs=1000

for option in 1 2 3 4 5
do
    random_seq_min_number=$(((option-1)*(option-1)))
        for vector in 1 2 3 4 5 6 7 8 9 10
        do
            sh ./shell/cem_hipo_template.sh ${task_name} ${env_scale} ${seed} ${option} ${vector} ${num_demo_per_program} ${cuda_devices} ${random_seq_min_number} ${max_number_of_epochs}
        done
done
