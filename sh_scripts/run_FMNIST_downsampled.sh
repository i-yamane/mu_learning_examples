set -eu

export MLFLOW_TRACKING_URI=file:./results_FashionMNIST; 

N_PARALELL=2

i=0
for ((seed=0; seed<2; ++seed)); do
    echo $seed
    python bench_experiment1.py \
        --model_name_h resnet20_prob_square \
        --model_name_f resnet20_prob_square \
        --model_name_u2y resnet20_prob_square \
        --n0 10000 \
        --n1 10000 \
        --epochs 200 \
        --exp_type bench_assump_violated \
        --transform downsampling \
        --downsampling_kernel 2 \
        --downsampling_stride 2 \
        --base_path ./data \
        --dataname FashionMNIST \
        --warm_start \
        --optimizer AdamW \
        --learning_rate 0.001 \
        --grad_clip 1e+15 \
        --weight_decay 0 \
        --n_channels 1 \
        --gpu_id 0 \
        --seed $seed \
        --no-show_plot \
        --mlflow_uri $MLFLOW_TRACKING_URI &

    i=$((i + 1))
    if (( i % $N_PARALELL == 0 )); then
        wait
        i=0
    fi
done

wait

N_PARALELL=3

i=0
for ((seed=2; seed<50; ++seed)); do
    echo $seed
# for seed in 0 2 3 4; do
    python bench_experiment1.py \
        --model_name_h resnet20_prob_square \
        --model_name_f resnet20_prob_square \
        --model_name_u2y resnet20_prob_square \
        --n0 10000 \
        --n1 10000 \
        --epochs 200 \
        --exp_type bench_assump_violated \
        --transform downsampling \
        --downsampling_kernel 2 \
        --downsampling_stride 2 \
        --base_path ~/data \
        --dataname FashionMNIST \
        --warm_start \
        --optimizer AdamW \
        --learning_rate 0.001 \
        --grad_clip 1e+15 \
        --weight_decay 0 \
        --gpu_id 0 \
        --n_channels 1 \
        --seed $seed \
        --no-show_plot \
        --mlflow_uri $MLFLOW_TRACKING_URI &

    i=$((i + 1))
    if (( i % $N_PARALELL == 0 )); then
        wait
        i=0
    fi
done

