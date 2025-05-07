from PIL import Image
import os

def create_patches(height, width, patch_size, image_open, image_name, output_path, patch_id):
    patch_ids = 0
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            box = (left, top, left + patch_size, top + patch_size)
            patch = image_open.crop(box)
            
            patch_filename = f"{image_name}_patch_{patch_id}_{patch_ids}.png"
            patch.save(os.path.join(output_path, patch_filename))
            patch_ids += 1

def pathes(orginal_image_folder, mask_image_path, output_path_real, output_path_mask, patch_size):
    os.makedirs(output_path_real, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)
    patch_id = 0
    for i in os.listdir(orginal_image_folder):
        i_mask = i.replace('.jpg', '_Mask.jpg')
        real_image = os.path.join(orginal_image_folder, i)
        mask_image = os.path.join(mask_image_path, i_mask)
        
        # Check if the mask image exists
        if not os.path.exists(mask_image):
            print(f"Mask image not found for {i}, skipping...")
            continue
        
        real_image_open = Image.open(real_image)
        mask_image_open = Image.open(mask_image)
        width_real, height_real = real_image_open.size
        width_mask, height_mask = mask_image_open.size
        create_patches(height_real, width_real, patch_size, real_image_open, i, output_path_real, patch_id)
        create_patches(height_mask, width_mask, patch_size, mask_image_open, i_mask, output_path_mask, patch_id)
        patch_id += 1
