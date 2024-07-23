import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
np.random.seed(0)

class ProbmapsAndScansDataset(Dataset):
    def __init__(self, args, dataset_dir, is_train=True):
        self.args = args
        self.data_dir = dataset_dir
        self.data = []
        self.optimize_csf = args.optimize_csf
        self.optimize_gm = args.optimize_gm
        self.optimize_wm = args.optimize_wm
        self.is_train = is_train
        self._setup_data()

    def _setup_data(self):
        self.scans_dir = os.path.join(self.data_dir, "scans")
        self.prob_maps_dir = os.path.join(self.data_dir, "probability_maps")
        if self.is_train:
            data_dir = os.path.join(self.prob_maps_dir, "train")
        else:
            data_dir = os.path.join(self.prob_maps_dir, "test")
        self.data = [os.path.join(data_dir, filename)
                        for filename in os.listdir(data_dir)]
        if self.is_train:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phantom = np.load(self.data[idx])

        # Discrete: (64, 64)
        gt_crisp = phantom['discrete'].squeeze()
        discrete = None
        output_phantom = None
        if self.args.optimize_csf:
            output_phantom = np.expand_dims(phantom['tissue_CSF'].squeeze(), 0)
            discrete = np.expand_dims(
                np.array(gt_crisp == 1, dtype=np.int32), 0
            )
        if self.args.optimize_gm:
            if isinstance(output_phantom, np.ndarray):
                output_phantom = np.concatenate(
                    [output_phantom, 
                     np.expand_dims(phantom['tissue_GM'].squeeze(), 0)]
                )
                discrete = np.concatenate(
                    [discrete, np.expand_dims(
                        np.array(gt_crisp == 2, dtype=np.int32), 0
                    )]
                )
            else:
                output_phantom = np.expand_dims(
                    phantom['tissue_GM'].squeeze(), 0
                )
                discrete = np.expand_dims(
                    np.array(gt_crisp == 2, dtype=np.int32), 0
                )
        if self.args.optimize_wm:
            if isinstance(output_phantom, np.ndarray):
                output_phantom = np.concatenate(
                    [output_phantom,
                     np.expand_dims(phantom['tissue_WM'].squeeze(), 0)]
                )
                discrete = np.concatenate(
                    [discrete, np.expand_dims(
                        np.array(gt_crisp == 3, dtype=np.int32), 0
                    )]
                )
            else:
                output_phantom = np.expand_dims(
                    phantom['tissue_WM'].squeeze(), 0
                )
                discrete = np.expand_dims(
                    np.array(gt_crisp == 3, dtype=np.int32), 0
                )

        # if self.is_train:
        input_image_space = None
        subject_id = os.path.split(self.data[idx])[1].replace(".npz", ".pt")
        for seq_idx in range(self.args.num_sequences):
            if self.is_train:
                scan_filepath = os.path.join(
                    self.scans_dir, "train", f"seq_{seq_idx}_{subject_id}"
                )
            else:
                scan_filepath = os.path.join(
                    self.scans_dir, "test", f"seq_{seq_idx}_{subject_id}"
                )
            if isinstance(input_image_space, type(None)):
                input_image_space = torch.load(scan_filepath)
            else:
                input_image_space = torch.concatenate([
                    input_image_space, torch.load(scan_filepath)
                ])

        return input_image_space, output_phantom, discrete, self.data[idx]