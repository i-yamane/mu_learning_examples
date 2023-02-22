set -eu

export MLFLOW_TRACKING_URI=file:./results_Breg_toy

N=$1

for ((seed=0; seed<5; ++seed)); do
    python toy_experiment_Breg1.py \
      --n0 $N \
      --n1 $N \
      --xdim 10\
      --noise_level_y 0.001\
      --noise_level_u 0.5\
      --epochs 1000\
      --d_hidden 50\
      --exp_type toy_regression1\
      --warm_start\
      --optimizer Adam\
      --learning_rate 0.001\
      --grad_clip 1E+15\
      --seed $seed\
      --no-show_plot\
      --mlflow_uri $MLFLOW_TRACKING_URI
done
