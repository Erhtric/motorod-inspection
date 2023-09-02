import argparse
import cv2
from pathlib import Path

from src.rod_detection import detect_rods_blob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motorod inspection.')
    parser.add_argument('--task', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="ispezione-bielle-immagini")
    parser.add_argument('--visualize', type=bool, default=True)


    args = parser.parse_args()

    data_folder = Path(args.data_dir)

    if args.task == 1:
        task_names = [
            "TESI00.BMP",
            "TESI01.BMP",
            "TESI12.BMP",
            "TESI21.BMP",
            "TESI31.BMP",
            "TESI33.BMP",
        ]
    elif args.task == 2:
        task_names = [
            "TESI44.BMP",
            "TESI47.BMP",
            "TESI48.BMP",
            "TESI49.BMP",
            "TESI50.BMP",
            "TESI51.BMP",
            "TESI90.BMP",
            "TESI92.BMP",
            "TESI98.BMP",
        ]
    else:
        raise ValueError("Task not supported.")
    
    paths = []
    for _path in data_folder.rglob("*.BMP"):
        paths.append(_path)

    images = [cv2.imread(str(_path), cv2.IMREAD_GRAYSCALE) for _path in paths]
    results = {}
    for i in range(6):
        if args.task == 1:
            results[i] = detect_rods_blob(images[i], visualize=True)
        elif args.task == 2:
            results[i] = detect_rods_blob(images[i], visualize=True, min_area=1500, detect_contact_pts=True)

    print("Finished.")