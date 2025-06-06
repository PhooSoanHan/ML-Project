import os
import random
from PIL import Image
import uuid

dataset_dir = "C:/Users/Phoo Soan Han/ML-Project/garbage-classification/resized_dataset"
output_dir = "C:/Users/Phoo Soan Han/ML-Project/garbage-classification/composite_images"
image_size = (128, 128)
composite_size = (384,384)

os.makedirs(output_dir, exist_ok = True)

# Load paths
classes = os.listdir(dataset_dir)
class_to_id = {cls: i for i, cls in enumerate(classes)}

def get_random_images(n):
    images = []
    labels = []
    for _ in range(n):
        cls = random.choice(classes)
        img_file = random.choice(os.listdir(os.path.join(dataset_dir, cls)))
        img_path = os.path.join(dataset_dir, cls, img_file)
        images.append((Image.open(img_path).resize(image_size), cls))
    return images

def create_composite_image():
    img_count = random.randint(2, 4)
    items = get_random_images(img_count)

    canvas = Image.new("RGB", composite_size, (255, 255, 255))
    positions = [(0, 0), (128, 0), (256, 0),
                 (0, 128), (128, 128), (256, 128),
                 (0, 256), (128, 256), (256, 256)]
    used_pos = random.sample(positions, img_count)

    annotations = []
    for (img, cls), (x, y) in zip(items, used_pos):
        canvas.paste(img, (x, y))
        annotations.append({
            "class": cls,
            "bbox": [x, y, x + image_size[0], y + image_size[1]]
        })

    filename = str(uuid.uuid4()) + ".jpg"
    canvas.save(os.path.join(output_dir, filename))

    with open(os.path.join(output_dir, filename.replace(".jpg", ".txt")), "w") as f:
        for ann in annotations:
            cls_id = class_to_id[ann["class"]]
            x_min, y_min, x_max, y_max = ann["bbox"]
            f.write(f"{cls_id} {x_min} {y_min} {x_max} {y_max}\n")

for _ in range(100):
    create_composite_image()

print("Composite images generated in:", output_dir)