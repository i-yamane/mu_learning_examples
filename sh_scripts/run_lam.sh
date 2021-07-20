set -eu

export MLFLOW_TRACKING_URI='http://localhost:6015'
# DIM=$1

i=0
for ((seed=0; seed<100; ++seed)); do
    # For Toy data
    # python toy_experiment1.py --epochs 100 --exp_type toy_assump_satisfied --xdim $DIM --udim $DIM --optimizer AdaDelta --seed $seed --labels v0.0.2 --tags host:`hostname`+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI &
    # python toy_experiment1.py --epochs 100 --exp_type toy_assump_satisfied --xdim $DIM --udim $DIM --optimizer AdaDelta --seed $seed --labels v0.0.2 --tags host:`hostname`+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI &
    # python toy_experiment1.py --epochs 100 --exp_type toy_assump_satisfied_xdim_fixed --n0 1000 --n1 1000 --xdim $DIM --udim $DIM --optimizer Adadelta --learning_rate 1 --seed $seed --labels v0.0.2+random_proj --tags host:`hostname`+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI &

    # For benchmark data
    # python bench_experiment1.py --n0 10000 --n1 10000 --epochs 100 --exp_type bench_assump_violated --udim $DIM --base_path ~/data --dataname CIFAR10 --optimizer Adam --seed $seed --labels bench_branch --tags host:ka10+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI &
    python bench_experiment1.py --n0 10000 --n1 10000 --epochs 100 --exp_type bench_assump_violated --transform downsampling --downsampling_kernel 2 --downsampling_stride 2 --base_path ~/data --dataname CIFAR10 --optimizer Adam --learning_rate 0.001 --seed $seed --labels bench_branch+dataaug+icml --tags bn:False --no-show_plot --mlflow_uri $MLFLOW_TRACKING_URI &
    # python bench_experiment1.py --n0 5000 --n1 5000 --epochs 100 --exp_type bench_assump_violated --dataname CIFAR10 --transform cropping --model_name resnet20 --optimizer Adam --base_path ~/data --seed $seed --labels resnet+crop --tags host:`hostname`+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI
    # python bench_experiment1.py --n0 5000 --n1 5000 --epochs 100 --exp_type bench_assump_violated --udim $DIM --base_path ~/data --dataname MNIST --optimizer AdaDelta --seed $seed --labels bench_branch --tags host:b6+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI &

    # python toy_experiment1.py --epochs 100 --exp_type toy_assump_violated_rand_proj --n0 1000 --n1 1000 --xdim 10 --udim $DIM --optimizer Adadelta --learning_rate 1 --seed $seed --labels random_proj+fixed_bug3 --tags host:`hostname`+bn:False --show_plot False --mlflow_uri $MLFLOW_TRACKING_URI

    n_paralell=1
    i=$((i + 1))
    if (( i % $n_paralell == 0 )); then
        wait
        i=0
    fi
done

