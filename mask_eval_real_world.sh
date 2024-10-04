#!/bin/bash

# 定义场景和物体列表
scenes=("food" "toys" "daily_necessities" "sundries")
objects_food=("apple" "orange" "banana" "mineral water bottle" "milk" "yogurt" "ham")
objects_toys=("ball" "doll" "car" "puzzle" "lego" "doll" "shovel" "bucket")
objects_daily_necessities=("toothbrush" "soap" "towel" "mug" "spoon" “toothpaste” "ball of yarn")
objects_sundries=("apple" "orange" "red box" "glasses" "mahjong" "toothpaste" "blue cup" "shovel" "bucket" "ball of yarn" "doll" "spoon")


# 提示用户选择场景
echo "请选择一个场景：${scenes[@]}"
read -p "输入场景: " user_scene

# 检查用户输入的场景是否有效
if [[ ! " ${scenes[@]} " =~ " ${user_scene} " ]]; then
    echo "无效的场景输入，请重新运行脚本并选择有效场景。"
    exit 1
fi

# 根据用户选择的场景设置物体列表
case $user_scene in
  food)
    objects=("${objects_food[@]}")
    ;;
  toys)
    objects=("${objects_toys[@]}")
    ;;
  daily_necessities)
    objects=("${objects_daily_necessities[@]}")
    ;;
  sundries)
    objects=("${objects_sundries[@]}")
    ;;
esac

# 随机抽取三个物体
selected_objects=($(shuf -e "${objects[@]}" -n 3))

# 输出选中的物体，并运行 Python 脚本
echo "随机抽取的三个物体是: ${selected_objects[@]}"
for object in "${selected_objects[@]}"; do
  echo "Running for scene: $user_scene, object: $object"
  python main.py --image_path ./data/images/ --mask_path ./data/masks/ --track_len 2 --prompt "$object" --mask_experiments
done

