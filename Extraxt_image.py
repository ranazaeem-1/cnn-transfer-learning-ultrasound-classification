import cv2
import os
import numpy as np

# Input folder (your extracted images)
input_dir = "extracted_images"

# Output folder for rotated & flipped images
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# Loop through all images
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):

        # Load the image
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # 1️⃣ Rotate 90 degrees clockwise
        # cv2.rotate rotates counterclockwise by default, so use ROTATE_90_CLOCKWISE
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # 2️⃣ Flip horizontally
        # flipCode = 1 → horizontal flip
        final_img = cv2.flip(rotated, 1)

        # Save output
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, final_img)

print("✔ Finished rotating and flipping all images!")
