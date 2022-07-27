import argparse
import glob
import os
import json

from tqdm import tqdm
import cv2
import numpy as np

from utils import gen_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labelme's file.")
    parser.add_argument("module", choices=["yolo", "ryolo"], help="Choice module.")
    parser.add_argument("-r", "--root_dir", help="The directory of images that need to be predicted.")
    args, setting = parser.parse_known_args()

    exec(f"from {args.module} import Model")
    predict = Model(setting)

    file_list = glob.glob(f'{args.root_dir}/**/*', recursive=True)
    image_list = [item for item in file_list if item.endswith(('.bmp', '.png', 'jpeg', 'jpg'))]

    for image_path in tqdm(image_list):
        image_dir, image_name = os.path.split(image_path)
        json_name = os.path.splitext(image_name)[0] + '.json'
        json_path = os.path.join(image_dir, json_name)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)

        flags, shapes = predict(image)
        data = gen_data(image_name, image.shape[0], image.shape[1], flags, shapes)

        json.dump(data, open(json_path, 'w', encoding='utf8'))


