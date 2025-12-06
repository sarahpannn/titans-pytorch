# e.g. 8 CPU cores, 32 GB RAM, 4 hours
salloc -p mit_normal_gpu \
    --gres=gpu:h200:1 \
    -c 8 \
    --mem=64G \
    --time=02:00:00