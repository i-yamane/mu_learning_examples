set -eu

export MLFLOW_TRACKING_URI=file:./results_Breg_MNIST

for ((seed=0; seed<5; ++seed)); do
    echo "seed: $seed"
    python bench_experiment_Breg1.py \
        --n0 10000 \
        --n1 10000 \
        --epochs 200 \
        --transform downsampling \
        --downsampling_kernel 2 \
        --downsampling_stride 2 \
        --base_path ./data \
        --dataname MNIST \
        --warm_start \
        --optimizer Adam \
        --learning_rate 0.001 \
        --grad_clip 1e+15 \
        --weight_decay 0 \
        --seed $seed \
        --n_channels 1 \
        --no-show_plot \
        --gpu_id 1 \
        --mlflow_uri $MLFLOW_TRACKING_URI
done
