import cv2
import numpy as np
import cv2
import numpy as np

# def get_mask_center(mask):
#     coords = np.column_stack(np.nonzero(mask))

#     if coords.shape[0] == 0:
#         return None
    
#     center = coords.mean(axis=0)
#     return int(round(center[1])), int(round(center[0]))

# def visualize_mask_center(mask, image_with_mask, center):
#     if center is not None:
#         # 在图像上画一个红色的圆圈表示中心点
#         cv2.circle(image_with_mask, center, radius=5, color=(0, 0, 255), thickness=-1)
#     return image_with_mask

# def process_single_image(image_path, output_path):
#     # 读取并转换图像为单通道灰度掩码
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mask = np.where(gray_image > 0, 1, 0).astype(np.uint8)

#     # 获取掩码的中心点
#     center = get_mask_center(mask)
    
#     # 可视化中心点
#     if center:
#         image_with_center = visualize_mask_center(mask, image, center)
#         cv2.imwrite(output_path, image_with_center)
#         print(f'Saved image with center visualized at {output_path}')
#     else:
#         print(f'No center found, skipping.')


def load_and_convert_mask(image_path):
    # 从文件读取彩色图像
    image = cv2.imread(image_path)
    
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 将非零部分转换为1，其他部分保持为0
    mask = np.where(gray_image > 0, 1, 0).astype(np.uint8)
    
    return mask

def get_mask_bbox(mask):
    coords = np.column_stack(np.nonzero(mask))
    
    if coords.shape[0] == 0:
        return None
    
    min = coords.min(axis=0)
    max = coords.max(axis=0)
    x_min, y_min = min[1], min[0]
    x_max, y_max = max[1], max[0]

    return int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))

def draw_bounding_box_on_mask(mask, bbox):
    x_min, y_min, x_max, y_max = bbox
    
    # 将box画到mask上，框的颜色是白色
    mask_with_box = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return mask_with_box

def process_single_image(image_path, output_path):
    # 转换图像为单通道灰度掩码
    mask = load_and_convert_mask(image_path)
    
    # 获取掩码的边界框
    bbox = get_mask_bbox(mask)
    
    if bbox:
        # 在掩码上绘制边界框
        mask_with_box = draw_bounding_box_on_mask(mask, bbox)
        
        # 保存结果图像
        cv2.imwrite(output_path, mask_with_box)
        print(f'Saved mask with bounding box at {output_path}')
    else:
        print(f'No bounding box found, skipping.')


if __name__ == "__main__":
    input_image_path = '/home/descfly/6d_pose/mast3r-grasp/data/masks/best_mask_20.png'
    output_image_path = './output_mask_with_box.png'

    process_single_image(input_image_path, output_image_path)

