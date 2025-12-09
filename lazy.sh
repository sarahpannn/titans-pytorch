# e.g. 8 CPU cores, 32 GB RAM, 4 hours
salloc -p mit_normal_gpu \
  --gres=gpu:h200:2 \
  -c 8 \
  --mem=64G \
  --time=05:00:00
