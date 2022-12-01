import os

from os.path import join, exists
from src.utils.preprocess_utils import preprocess_widerface, merge_yolo


if __name__ == "__main__":
    # root for yolo format dataset
    yolo_root = './datasets/yolo'
    # Preprocess Wider Face dataset
    widerface_raw_root = './datasets/widerface'
    preprocess_widerface(raw_root=widerface_raw_root, processed_root=yolo_root, remove_processed=True, keep_dir=True)
    merge_yolo('./datasets/other_obj', yolo_root)

    f = open(join(yolo_root, 'meta.yaml'), 'w')
    train_image_dir = join(yolo_root, 'images/train')
    val_image_dir = join(yolo_root, 'images/val')
    test_image_dir = join(yolo_root, 'images/test')
    f.write(f'train: {train_image_dir}')
    f.write(f'\nval: {val_image_dir}')
    f.write(f'\ntest: {test_image_dir}')
    f.write(f'\nnc: {4}')
    f.write("\nnames: ['Face', 'Clen', 'Cam', 'Mobile']")