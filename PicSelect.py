import os
import random
import shutil

def select_random_images(source_folder, num_images):
    images = os.listdir(source_folder)
    random_images = random.sample(images, num_images)
    return random_images

def copy_images_to_destination(images, source_folder, destination_folder):
    for image in images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.copy(source_path, destination_path)

def main():
    source_folder = 'data/images/train'
    destination_folder1 = 'SamarthPics'
    destination_folder2 = 'AyushPics'
    num_images_to_select = 500

    # Select random images
    random_images_set1 = select_random_images(source_folder, num_images_to_select)
    random_images_set2 = select_random_images(source_folder, num_images_to_select)

    # Copy images to destination folders
    copy_images_to_destination(random_images_set1, source_folder, destination_folder1)
    copy_images_to_destination(random_images_set2, source_folder, destination_folder2)

if __name__ == "__main__":
    main()
