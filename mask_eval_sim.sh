#!/bin/bash
for i in $(seq 1 1)
do
  echo "Iteration $i"
  python evaluation/mask_evaluation.py --image_path ./data/images/ --mask_path ./data/masks/ --track_len 2
done

