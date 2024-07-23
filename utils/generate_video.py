import cv2
import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Video generator script")
    parser.add_argument("--exp", "-e", type=str, required=True,
                        help="Path to the experiments directory")
    return parser.parse_args()

def main(args):
    plots_dir = os.path.join(args.exp, "plots")
    video_path = os.path.join(args.exp, "videos", "out.avi")

    epochs_dirs = [
        (os.path.join(plots_dir, dir), int(dir.split("_")[1])) for dir in os.listdir(plots_dir) \
        if dir.startswith("epoch")
    ]

    epochs_dirs =  sorted(epochs_dirs, key=lambda x: x[1])

    epochs_dirs = [dir[0] for dir in epochs_dirs]

    height, width, _ = cv2.imread(os.path.join(epochs_dirs[0], "csf.png")).shape

    total = 0.5
    height_per_image = int(height * total)
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc('M','J','P','G'),
        5,
        (width, height_per_image * 3)
    )

    print(f"Writing video at {video_path} of shape ({height_per_image*3}, {width}, 3)")
    for epoch_id, dir in enumerate(epochs_dirs):
        print(f"Writing video of epoch: {dir}")
        csf = cv2.imread(os.path.join(dir, "csf.png"))
        gm = cv2.imread(os.path.join(dir, "gm.png"))
        wm = cv2.imread(os.path.join(dir, "wm.png"))

        image = np.zeros([int(height*total)*3, width, 3], dtype=np.uint8)

        start_y = int(height * (1-total)/2)
        end_y = start_y + height_per_image
        image[0:height_per_image, :, :] = csf[start_y:end_y, :, :]
        image[height_per_image:height_per_image*2, :, :] = gm[start_y:end_y, :, :]
        image[height_per_image*2:, :, :] = wm[start_y:end_y, :, :]

        writer.write(image)

if __name__ == "__main__":
    main(parse_args())