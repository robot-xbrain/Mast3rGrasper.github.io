import cv2  # type: ignore
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import os
import numpy as np
import torch
from PIL import Image
import open_clip
import time  # 用于计时

# my_own_class_names = ['an apple', 'a banana', 'an orange', 'a lemon', 'cucumber', 'carrot', 
#                       'glasses', 'cigarette box', 'toothpaste', 'a red box', 'a pencil',
#                       'shovel', 'spoon', 'mahjong', 'a blue cup']
my_own_class_names = ['an apple', 'a banana', 'an orange', 'a lemon', 'cucumber', 'carrot', 
                      'glasses', 'cigarette box', 'toothpaste', 'a red box', 'a pencil',
                      'shovel', 'spoon', 'mahjong']

class_names = ['a phone', 'a house']
parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="default",
    help="The type of model to load, in ['default', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
parser.add_argument("--text", type=str, help="The text prompt to compare with the masks.")

def apply_mask(image, mask):
    """Apply the given mask to the image and set other regions to black."""
    masked_image = np.zeros_like(image)
    masked_image[mask == 1] = image[mask == 1]
    return masked_image.astype(np.uint8)

def BestMask(text, checkpoint, image_input, output, device = "cuda", save_other_mask=False) -> None:
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    clip_model.to(device)  
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text_list = [text] + class_names
    text_tokens = tokenizer(text_list).to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    image = cv2.imread(image_input)
    if image is None:
        print(f"Could not load '{image_input}' as an image, exiting...")
        return

    image_cp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image_cp)
    
    best_mask = None
    highest_similarity = -1
    best_mask_index = -1
    if save_other_mask:
        mask_output_dir = os.path.join(output, "masks")
        os.makedirs(mask_output_dir, exist_ok=True)

    for idx, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        mask = mask > 0.1

        masked_image = apply_mask(image_cp, mask)

        if save_other_mask:
            mask_output_path = os.path.join(mask_output_dir, f"mask_{idx}.png")
            Image.fromarray(masked_image).save(mask_output_path)

        masked_image_pil = Image.fromarray(masked_image)
        masked_image_tensor = preprocess(masked_image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(masked_image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            value = similarity[0][0].item()  

        if value > highest_similarity:
            highest_similarity = value
            best_mask = masked_image
            best_mask_index = idx

    if best_mask is not None:
        print(f"Highest similarity: {highest_similarity:.2f}")
        output_path = os.path.join(output, f"mask_0_{text}.png")
        Image.fromarray(best_mask).save(output_path)
    del sam, clip_model, mask_generator, text_features, text_tokens
    return masks[best_mask_index]['segmentation']

if __name__ == "__main__":
    args = parser.parse_args()
    best_mask = BestMask(args)
    best_mask_img = Image.fromarray(best_mask)
    best_mask_img.show()
    time.sleep(3)
    best_mask_img.close()
