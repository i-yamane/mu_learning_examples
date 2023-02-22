set -eu

export MLFLOW_TRACKING_URI=file:./results_CIFAR100

for ((seed=0; seed<5; ++seed)); do
    echo "seed: $seed"
    python bench_experiment1.py \
        --model_name_h resnet20_prob_square \
        --model_name_f resnet20_prob_square \
        --model_name_u2y resnet20_prob_square \
        --n0 10000 \
        --n1 10000 \
        --epochs 200 \
        --transform downsampling \
        --downsampling_kernel 2 \
        --downsampling_stride 2 \
        --base_path ./data \
        --dataname CIFAR100 \
        --warm_start \
        --optimizer AdamW \
        --learning_rate 0.001 \
        --grad_clip 1e+15 \
        --weight_decay 0 \
        --n_channels 3 \
        --gpu_id 0 \
        --seed $seed \
        --mlflow_uri $MLFLOW_TRACKING_URI
done

