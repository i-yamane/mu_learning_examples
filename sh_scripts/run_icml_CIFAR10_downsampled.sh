set -eu

export MLFLOW_TRACKING_URI='http://localhost:6415'
# DIM=$1

i=0
for ((seed=0; seed<50; ++seed)); do
    python bench_experiment1.py --n0 10000 --n1 10000 --epochs 200 --exp_type bench_assump_violated --transform downsampling --downsampling_kernel 2 --downsampling_stride 2 --base_path ~/data --dataname CIFAR10 --warm_start --optimizer Adam --learning_rate 0.01 --grad_clip 1E+15 --seed $seed --labels bench_branch+dataaug+icml+no_grad_clip --tags bn:False --no-show_plot --mlflow_uri $MLFLOW_TRACKING_URI &

    n_paralell=3
    i=$((i + 1))
    if (( i % $n_paralell == 0 )); then
        wait
        i=0
    fi
done

