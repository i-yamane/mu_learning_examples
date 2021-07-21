set -eu

export MLFLOW_TRACKING_URI=file:./results_toy 

for ((seed=0; seed<5; ++seed)); do
    # For Toy data
    for DIM in 2 5 10 15 20; do
      python toy_experiment1.py --epochs 200 --exp_type toy_assump_satisfied_ctl_noise --n0 1000 --n1 1000 --xdim $DIM --udim $DIM --noise_level_u=0.5 --optimizer Adam --learning_rate 0.01 --seed $seed --mlflow_uri $MLFLOW_TRACKING_URI
      python toy_experiment1.py --epochs 200 --exp_type toy_assump_violated_ctl_noise --n0 1000 --n1 1000 --xdim $DIM --udim $DIM --noise_level_u=0.5 --optimizer Adam --learning_rate 0.01 --seed $seed --mlflow_uri $MLFLOW_TRACKING_URI
    done
done

