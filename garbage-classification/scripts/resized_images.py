import os
import cv2

input_dir = "../dataset/raw"
output_dir = "../resized_dataset"
target_size = (128, 128) #Targeted size to resize images

#Create output_dir if its not exist
os.makedirs(output_dir, exist_ok = True)

for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)

    # To make sure its a folder
    if os.path.isdir(class_input_path):
        os.makedirs(class_output_path, exist_ok=True)
        print(f"Resizing images in {class_name}...")

        for filename in os.listdir(class_input_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_input_path, filename)
                img = cv2.imread(img_path)

                if img is None:
                    print(f" Skipping unreadable image: {img_path}")
                    continue

                # Resize image
                resized_img = cv2.resize(img, target_size)

                # Save the image
                output_path = os.path.join(class_output_path, filename)
                cv2.imwrite(output_path, resized_img)

print("All images resized!!!")