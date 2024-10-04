import cv2
import numpy as np
import torch
import time
import pyrealsense2 as rs
from sam2.build_sam import build_sam2_camera_predictor
import os
from collections import defaultdict
from glob import glob
import re
import sys
sys.path.append("./get_mask_track")
from get_first_mask.amp_points import BestMask


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

class MaskSaver:
    def __init__(self):
        self.index = 1

    def save(self, mask, obj_index, text, save_dir):
        for i in range(0, len(obj_index)):
            filename = os.path.join(save_dir, f"mask_{self.index}_{text}.png")
            cv2.imwrite(filename, mask)
            # print(f"Saved mask {filename}")
        self.index += 1

class BoxSaver:
    def __init__(self):
        self.index = 2
    def save(self, mask, obj_index, text, save_dir):
        for i in range(0, len(obj_index)):
            filename = os.path.join(save_dir, f"box_{self.index}_{text}.txt")
            mask[i] = mask[i] / 255
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            data = [x_min, y_min, x_max, y_max]
            with open(filename, 'w') as file:
                for value in data:
                    file.write(f"{value}\n")
            # print(f"Saved box {filename}")
        self.index += 1

def convert_property(property_value):
    return 1 if property_value == 'Positive' else 0

def get_mask_bbox(mask):
    coords = np.column_stack(np.nonzero(mask))
    
    if coords.shape[0] == 0:
        return None
    
    min = coords.min(axis=0)
    max = coords.max(axis=0)
    x_min, y_min = min[1], min[0]
    x_max, y_max = max[1], max[0]

    return int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))

def get_mask_center(mask):
    coords = np.column_stack(np.nonzero(mask))

    if coords.shape[0] == 0:
        return None
    
    center = coords.mean(axis=0)
    return int(round(center[1])), int(round(center[0]))

def mask_tracker(dataset_path, save_dir, text, image_len=20, save_type="mask"):

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

    if_init = False
    if save_type == "mask":
        saver = MaskSaver()
    elif save_type == "box":
        saver = BoxSaver()

    read_images = []
    pair_len = image_len
    sam1_path = "./checkpoints/sam_vit_h_4b8939.pth"
    mask_list = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            if len(read_images) >= pair_len:
                break
            all_files = sorted(glob(f'{dataset_path}/*'), key=natural_sort_key)
            new_files = [f for f in all_files if f not in read_images]
            if len(new_files) >= 1:
                read_images.extend(new_files[:1])
                frame = cv2.imread(read_images[-1])

                if not if_init:
                    first_mask = BestMask(text=text, checkpoint=sam1_path, image_input=read_images[-1], 
                                          output=save_dir, device = "cuda")  
                    height = frame.shape[0]
                    width = frame.shape[1]
                    predictor.load_first_frame(frame)
                    ann_frame_idx = 0
                    if_init = True
                    ann_obj_id = 0

                    x, y = get_mask_center(first_mask)
                    point = np.array([[x, y]], np.float32)
                    labels = np.array([1], np.int32)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=point,
                        labels=labels,
                    )
                    mask_list.append(first_mask)

                    # box = get_mask_bbox(first_mask)

                    # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    #     frame_idx=ann_frame_idx,
                    #     obj_id=ann_obj_id,
                    #     bbox=box,
                    # )

                else:
                    out_obj_ids, out_mask_logits = predictor.track(frame)
                    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                    for i in range(0, len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                            np.uint8
                        ) * 255
                        all_mask = cv2.bitwise_or(all_mask, out_mask)
                    mask_list.append(all_mask)
                    saver.save(all_mask, out_obj_ids, text, save_dir)
                    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
                    frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

                #cv2.imshow("frame", frame)
            else :
                print(f"当前读取了 {len(read_images)} 帧，等待更多文件...")
                # 等待一段时间再尝试读取
                time.sleep(1)
    
    return mask_list
    