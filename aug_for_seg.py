import cv2
import numpy as np
import os
import random

random.seed(67)

input_dirs = ["./data/data_seg/Cucumber/img", "./data/data_seg/Strawberry/img"]
output_dirs = ["./data/data_seg/Cucumber/mask", "./data/data_seg/Strawberry/mask"]
trash_dir = "./data/Trash"

for i in range(len(input_dirs)):
    input_dir = input_dirs[i]
    output_dir = output_dirs[i]

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # порог — можно подбирать
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        cv2.imwrite(os.path.join(output_dir, filename), mask)


def cut_object(img, mask):
    obj = cv2.bitwise_and(img, img, mask=mask)
    return obj

def random_transform(obj, mask):
    h, w = obj.shape[:2]
    
    # масштаб
    scale = random.uniform(1.25, 1.75)
    new_size = (int(w * scale), int(h * scale))
    
    obj = cv2.resize(obj, new_size)
    mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    
    # поворот
    angle = random.uniform(0, 360)
    center = (new_size[0] // 2, new_size[1] // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    obj = cv2.warpAffine(obj, M, new_size)
    mask = cv2.warpAffine(mask, M, new_size, flags=cv2.INTER_NEAREST)

    # отражение
    m = random.choice([-1, 0, 1, 2])
    if m != 2:
        obj = cv2.flip(obj, m)
        mask = cv2.flip(mask, m)
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    obj = cv2.GaussianBlur(obj, (3,3), 0)

    return obj, mask

def paste_object(bg, obj, mask):
    bh, bw = bg.shape[:2]
    oh, ow = obj.shape[:2]

    if ow >= bw or oh >= bh:
        return bg, mask, 0, 0

    x = random.randint(0, bw - ow)
    y = random.randint(0, bh - oh)

    roi = bg[y:y+oh, x:x+ow].astype(np.float32)
    obj = obj.astype(np.float32)

    # делаем мягкую маску
    mask_blur = cv2.GaussianBlur(mask, (7,7), 0)
    alpha = mask_blur / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    blended = alpha * obj + (1 - alpha) * roi

    bg[y:y+oh, x:x+ow] = blended.astype(np.uint8)

    return bg, (mask > 127).astype(np.uint8)*255, x, y

def update_mask(final_mask, obj_mask, x, y):
    oh, ow = obj_mask.shape
    
    roi = final_mask[y:y+oh, x:x+ow]
    
    final_mask[y:y+oh, x:x+ow] = np.maximum(roi, obj_mask)
    
    return final_mask

for ind in range(len(input_dirs)):
    img_dir = input_dirs[ind]
    mask_dir = output_dirs[ind]
    for i in range(5000):
        bg = cv2.imread(random.choice([os.path.join(trash_dir, f) for f in os.listdir(trash_dir)]))
        final_mask = np.zeros(bg.shape[:2], dtype=np.uint8)

        num_objects = random.randint(1, 5)

        for _ in range(num_objects):
            img_name = random.choice([name for name in os.listdir(img_dir)])

            filename = os.path.join(img_dir, img_name)   
            mask_path = os.path.join(mask_dir, img_name)

            img = cv2.imread(filename) 
            mask = cv2.imread(mask_path, 0)
    
            obj = cut_object(img, mask)
            obj, mask = random_transform(obj, mask)

            bg, mask, x, y = paste_object(bg, obj, mask)
            final_mask = update_mask(final_mask, mask, x, y)

        cv2.imwrite(f"./data/data_seg/cpaug/{img_dir[16:-4]}/img/{i}.jpg", bg)
        cv2.imwrite(f"./data/data_seg/cpaug/{mask_dir[16:-5]}/mask/{i}.jpg", final_mask)





