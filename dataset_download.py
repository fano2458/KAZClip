import json
import os
import requests
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def download_images(dataset, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img_info in tqdm(dataset["images"]):
        local_path = os.path.join(output_folder, img_info["file_name"])
        if os.path.exists(local_path):
            continue
        url = img_info["coco_url"]
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    print("Download complete!")


if __name__ == "__main__":
    val_dataset = load_json("data/val2017.json")
    download_images(val_dataset, "data/val2017")

    train_dataset = load_json("data/train2017.json")
    download_images(train_dataset, "data/train2017")