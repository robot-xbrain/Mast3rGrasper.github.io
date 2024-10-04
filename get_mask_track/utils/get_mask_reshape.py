import numpy as np
from PIL import Image
import cv2
import os

def resize_np_array(np_array, target_size):
    """
    Resize a numpy array representing an image to the target size.
    """
    img = Image.fromarray(np_array)
    S = max(img.size)
    if S > target_size:
        interp = Image.LANCZOS
    else:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * target_size / S)) for x in img.size)
    resized_img = img.resize(new_size, interp)
    return np.array(resized_img)

def crop_np_array(np_array, target_size, square_ok=False):
    """
    Crop a numpy array representing an image to the target size, center-cropped.
    """
    img = Image.fromarray(np_array)
    W, H = img.size
    cx, cy = W // 2, H // 2
    
    if target_size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = 3 * halfw // 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    
    return np.array(img)

def process_np_array(np_array, target_size, square_ok=False):
    """
    Resize and crop a numpy array representing an image.
    """
    resized_array = resize_np_array(np_array, target_size)
    cropped_array = crop_np_array(resized_array, target_size, square_ok)
    return cropped_array

def get_mask_reshape(mask_list):
    # save mask_list
    for mask in range(0, len(mask_list)):
        mask_path = os.path.join('results', "mask_" + str(mask) + ".png")
        cv2.imwrite(mask_path, (mask_list[mask].astype(np.uint8)) * 255)
    print('save masks done')

    for i in range(0, len(mask_list)):
        mask_list[i] = process_np_array(mask_list[i], 512)

    # save mask_list
    for mask in range(0, len(mask_list)):
        mask_path = os.path.join('results', "processed_mask_" + str(mask) + ".png")
        cv2.imwrite(mask_path, (mask_list[mask].astype(np.uint8)) * 255)
    print('save masks done')
    return mask_list

if __name__ == "__main__":
    w, h = 500, 300
    np_array = np.random.randint(0, 255, (w, h), dtype=np.uint8)

    # 处理这个 numpy 数组到目标尺寸
    target_size = 224
    processed_array = process_np_array(np_array, target_size)
    print(f"Processed array shape: {processed_array.shape}")
