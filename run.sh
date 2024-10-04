#!/bin/bash
for i in $(seq 1 1)
do
  echo "Iteration $i"
  python main.py --image_path ./data/images/ --mask_path ./data/masks/ --track_len 2 --prompt "a box with a pattern" --mask_experiments
  python grasper/anygrasp/grasp_detection/grasp_detector.py --top_down_grasp
  # cd grasper/real_robot_ws
  # source devel/setup.bash
  # rosrun grasp_dem ur5_grasp.py
done

# a bottle of water