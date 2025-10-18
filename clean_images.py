from PIL import Image
import os


def clean_jpegs(folder_path, overwrite=False):
    """
    Re-saves all JPEGs in folder to remove extraneous bytes.
    :param folder_path: path to the folder containing images
    :param overwrite: if True, overwrites original files; else saves with '_fixed' suffix
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.load()  # make sure image is loaded fully
                    if overwrite:
                        save_path = file_path
                    else:
                        name, ext = os.path.splitext(file_path)
                        save_path = f"{name}_fixed{ext}"
                    img.save(save_path)
                    print(f"Cleaned: {file_path} -> {save_path}")
                except Exception as e:
                    print(f"Skipped (cannot open): {file_path} | Reason: {e}")


# Example usage:
# clean_jpegs("./data/PetImages", overwrite=True)

import os


def find_empty_or_corrupt_images(folder_path):
    empty_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            path = os.path.join(root, f)
            if os.path.getsize(path) == 0:
                empty_files.append(path)
                os.remove(path)
    return empty_files


folder = "./data/PetImages"
bad_files = find_empty_or_corrupt_images(folder)
print("Empty files:", bad_files)
