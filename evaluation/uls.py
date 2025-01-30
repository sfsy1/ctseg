import zipfile, os


def unzip_files_in_dir(dir_path):
    dir_path = "/media/liushifeng/KINGSTON/ULS Jan 2025/ULS23/novel_data/ULS23_DeepLesion3D/labels/"
    for file in os.listdir(dir_path):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(dir_path, file), 'r') as zip_ref:
                zip_ref.extractall(dir_path)
            os.remove(os.path.join(dir_path, file))



def train_val_split(data_folder):
    label_names = {x.split(".")[0] for x in os.listdir(data_folder / "labels")}
    image_names = {x.split(".")[0] for x in os.listdir(data_folder / "images")}
    valid_names = label_names.intersection(image_names)
    print(f"{len(image_names)} images, {len(label_names)} labels, {len(valid_names)} both")

    # train/val split
    val_names = sorted(valid_names)[::4]
    train_names = [x for x in valid_names if x not in val_names]
    return train_names, val_names