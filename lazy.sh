# e.g. 8 CPU cores, 32 GB RAM, 4 hours
salloc -p mit_normal_gpu \
  --gres=gpu:l40s:1 \
  -c 8 \
  --mem=32G \
  --time=05:00:00 \
  --exclude=node3405
