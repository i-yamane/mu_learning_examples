set -eu

export MLFLOW_TRACKING_URI='http://localhost:6115'
# export MLFLOW_TRACKING_URI='sqlite:///results/debug_jan_30'

# DIM=$1

i=0
for ((seed=1; seed<50; ++seed)); do
    # For Toy data
    for DIM in 2 5 10 15 20; do
      python toy_experiment1.py --epochs 200 --exp_type toy_assump_satisfied_ctl_noise --n0 1000 --n1 1000 --xdim $DIM --udim $DIM --noise_level_u=0.5 --optimizer Adam --learning_rate 0.01 --seed $seed --labels v0.0.2+icml+no12 --tags host:`hostname`+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI &
      wait
      python toy_experiment1.py --epochs 200 --exp_type toy_assump_violated_ctl_noise --n0 1000 --n1 1000 --xdim $DIM --udim $DIM --noise_level_u=0.5 --optimizer Adam --learning_rate 0.01 --seed $seed --labels v0.0.2+icml+no12 --tags host:`hostname`+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI &
      wait
    done

    # 'toy_assump_satisfied'
    # 'toy_assump_violated'
    # 'toy_assump_satisfied_rand_proj'
    # 'toy_assump_violated_rand_proj'
    # 'toy_assump_satisfied_sin'
    # 'toy_assump_violated2'

    n_paralell=1
    i=$((i + 1))
    if (( i % $n_paralell == 0 )); then
        wait
        i=0
    fi
done

