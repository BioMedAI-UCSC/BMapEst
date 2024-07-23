import cv2
import os
import requests
import gzip
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
import torch

np.random.seed(0)
torch.manual_seed(0)

debug_res_dir = "./debug"
os.makedirs(debug_res_dir, exist_ok=True)
# Disable slow perlin noise
NO_NOISE = True
NOISE_TYPE = 'perlin' #alternative 'gaussian'

if not NO_NOISE:
    # Needs perlin_numpy: https://github.com/pvigier/perlin-numpy
    from perlin_numpy import generate_fractal_noise_3d, generate_perlin_noise_3d
# NOTE: tissues are blended linearly.
# Exp. fits might be better, but not in general and are more complicated

# The folder that contains this file and a cache of all tissues
BRAINWEB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "brainweb_raw_data")

# The BrainWeb data is centered in the MAP_SIZE^3 volume
MAP_SIZE = 432
DOWNSAMPLE_SIZE = 64
SAVE_MIDDLE_SLICE = True
DEBUG = True
NUM_SUBJECTS = 500
output_dir = f"../training_data/brainweb_{DOWNSAMPLE_SIZE}"
plots_dir = os.path.join(output_dir, "plots")

csf_plot_dir = os.path.join(plots_dir, "csfs")
gm_plot_dir = os.path.join(plots_dir, "gms")
wm_plot_dir = os.path.join(plots_dir, "wms")
discrete_plot_dir = os.path.join(plots_dir, "discretes")

os.makedirs(output_dir)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(BRAINWEB_PATH, exist_ok=True)
os.makedirs(csf_plot_dir, exist_ok=True)
os.makedirs(gm_plot_dir, exist_ok=True)
os.makedirs(wm_plot_dir, exist_ok=True)
os.makedirs(discrete_plot_dir, exist_ok=True)


# enumeration of all available BrainWeb tissues
class Tissue(IntEnum):
    CSF = 0
    GRAY_MATTER = 1
    WHITE_MATTER = 2
    FAT = 3
    MUSCLES = 4
    MUSCLES_SKIN = 5
    SKULL = 6
    VESSELS = 7
    CONNECTIVE = 8
    DURA = 9
    BONE_MARROW = 10
    DISCRETE = 11


TISSUE_DOWNLOAD_ALIAS = [
    "csf", "gry", "wht", "fat", "mus",
    "m-s", "skl", "ves", "fat2", "dura", "mrw", "crisp"
]

SUBJECTS = [
    4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
]

IGNORE_SUBJECTS = [sub for idx, sub in enumerate(SUBJECTS) if idx%2==0]

SUBJECTS = list(filter(lambda i: i not in IGNORE_SUBJECTS, SUBJECTS))

def draw(array, saveat, clim=(0, 1)):
        figure = plt.figure()
        plt.imshow(array, clim=clim)
        plt.savefig(saveat)
        plt.close(figure)

def generate_b0_b1_map_for_brainweb(PD):
    x_pos, y_pos, z_pos = torch.meshgrid(
        torch.linspace(-1, 1, PD.shape[0]),
        torch.linspace(-1, 1, PD.shape[1]),
        torch.linspace(-1, 1, PD.shape[2]),
        indexing="ij"
    )
    B1 = torch.exp(-(0.4*x_pos**2 + 0.2*y_pos**2 + 0.3*z_pos**2))
    dist2 = (0.4*x_pos**2 + 0.2*(y_pos - 0.7)**2 + 0.3*z_pos**2)
    B0 = 7 / (0.05 + dist2) - 45 / (0.3 + dist2)
    # Normalize such that the weighted average is 0 or 1
    weight = PD / PD.sum()
    B0 -= (B0 * weight).sum()
    B1 /= (B1 * weight).sum()
    return B0, B1[None, ...]


def load(subject: int, tissue: Tissue) -> np.ndarray:
    download_alias = f"subject{subject:02d}_{TISSUE_DOWNLOAD_ALIAS[tissue]}"
    file_name = download_alias + ".i8.gz"  # 8 bit signed int, gnuzip
    file_dir = os.path.join(BRAINWEB_PATH, f"subject{subject:02d}")
    file_path = os.path.join(file_dir, file_name)
    try:
        os.mkdir(file_dir)  # create the cache folder (if it doesn't exist)
    except FileExistsError:
        pass

    # If the file is not cached yet, we will download it
    if not os.path.exists(file_path):
        print(f"Couldn't find {file_name}, downloading it...")
        response = requests.post(
            "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1",
            data={
                "do_download_alias": download_alias,
                "format_value": "raw_byte",
                "zip_value": "gnuzip",
            }
        )

        with open(file_path, "wb") as f:
            f.write(response.content)

    # Now the file is guaranteed to exist so we can load it
    with gzip.open(file_path) as f:
        # BrainWeb states the data is unsigned, but that's plain wrong
        data = np.frombuffer(f.read(), np.uint8)

        # Don't add 128 for crisp
        if not tissue==Tissue.DISCRETE:
            data = data + 128

        # Coordinate system:
        #  - x points to right ear
        #  - y points to nose
        #  - z points to top of head
        # Indices are data[x, y, z]

        return data.reshape(362, 434, 362).swapaxes(0, 2)

tissue_GMs = []
tissue_WMs = []
tissue_CSFs = []
discretes = []

for subject in SUBJECTS:
    print(f"Loading subject: {subject}")
    discretes.append(load(subject, Tissue.DISCRETE))
    tissue_GMs.append(load(subject, Tissue.GRAY_MATTER))
    tissue_WMs.append(load(subject, Tissue.WHITE_MATTER))
    vessels = load(subject, Tissue.VESSELS)
    vessels[vessels > 0] -= 1
    tissue_CSFs.append(
        vessels + load(subject, Tissue.CSF)
    )

final_GM = []
final_WM = []
final_CSF = []

for tissue_GM, tissue_WM, tissue_CSF, discrete in zip(
    tissue_GMs, tissue_WMs, tissue_CSFs, discretes
):

    if DEBUG:
        brain_shape = tissue_GM.shape
        cv2.imwrite(os.path.join(debug_res_dir, "GM_Slice.png"), tissue_GM[:, :, brain_shape[2]//2])
        cv2.imwrite(os.path.join(debug_res_dir, "WM_Slice.png"), tissue_WM[:, :, brain_shape[2]//2])
        cv2.imwrite(os.path.join(debug_res_dir, "CSF_Slice.png"), tissue_CSF[:, :, brain_shape[2]//2])

    # Downsample and pad to get 128^3 maps
    print("Downsample and center maps")

    def downsample(tensor: np.ndarray):
        # tensor shape must be a multiple of 3 for this to work - remove excess
        shape = (np.array(tensor.shape) // 3) * 3
        tensor = tensor[:shape[0], :shape[1], :shape[2]].astype(np.float32)
        # tensor = tensor[0::3, :, :] + tensor[1::3, :, :] + tensor[2::3, :, :]
        # tensor = tensor[:, 0::3, :] + tensor[:, 1::3, :] + tensor[:, 2::3, :]
        # tensor = tensor[:, :, 0::3] + tensor[:, :, 1::3] + tensor[:, :, 2::3]
        return tensor 

    tissue_GM = downsample(tissue_GM)
    tissue_WM = downsample(tissue_WM)
    tissue_CSF = downsample(tissue_CSF)
    discrete = downsample(discrete)

    # Find the extends of the brain to center it
    total = tissue_GM + tissue_WM + tissue_CSF

    mask = total > 0.01
    test = mask.copy()
    x_indices, y_indices, z_indices = np.nonzero(mask)
    x_indices, y_indices, z_indices = np.where(total > 0.1)
    min_x = x_indices.min()
    max_x = x_indices.max() + 1
    min_y = y_indices.min()
    max_y = y_indices.max() + 1
    min_z = z_indices.min()
    max_z = z_indices.max() + 1

    # Warn if brain is too large
    length_x = max_x - min_x
    length_y = max_y - min_y
    length_z = max_z - min_z

    if length_x > MAP_SIZE or length_y > MAP_SIZE or length_z > MAP_SIZE:
        print(f"WARNING: Brain size = {length_x} x {length_y} x {length_z}")
        print(f"Maximum size is MAP_SIZE={MAP_SIZE}^3, maps will be truncated")

    # Center it (and cut to size if too large)
    length_x = min(MAP_SIZE, length_x)
    length_y = min(MAP_SIZE, length_y)
    length_z = min(MAP_SIZE, length_z)
    min_x = int((min_x + max_x) / 2 - length_x / 2)
    min_y = int((min_y + max_y) / 2 - length_y / 2)
    min_z = int((min_z + max_z) / 2 - length_z / 2)
    max_x = min_x + length_x
    max_y = min_y + length_y
    max_z = min_z + length_z

    def add_padding(data):
        # return data
        pad_x = (MAP_SIZE - length_x) // 2
        pad_y = (MAP_SIZE - length_y) // 2
        pad_z = (MAP_SIZE - length_z) // 2

        padded = np.zeros((MAP_SIZE, MAP_SIZE, MAP_SIZE), dtype=data.dtype)
        padded[
            pad_x:(pad_x+length_x),
            pad_y:(pad_y+length_y),
            pad_z:(pad_z+length_z)
        ] = data[min_x:max_x, min_y:max_y, min_z:max_z]
        return padded

    mask = add_padding(mask)
    total = add_padding(total)
    tissue_GM = add_padding(tissue_GM)
    tissue_WM = add_padding(tissue_WM)
    tissue_CSF = add_padding(tissue_CSF)
    discrete = add_padding(discrete)

    tissue_GM[mask] /= total[mask]
    tissue_WM[mask] /= total[mask]
    tissue_CSF[mask] /= total[mask]
    tissue_GM[~mask] = 0
    tissue_WM[~mask] = 0
    tissue_CSF[~mask] = 0

    final_CSF.append(tissue_CSF)
    final_GM.append(tissue_GM)
    final_WM.append(tissue_WM)
    
final_CSF = np.stack(final_CSF)
final_GM = np.stack(final_GM)
final_WM = np.stack(final_WM)

slice_idx = final_CSF.shape[-1]//2
pixelwise_mean_csf = np.expand_dims(np.mean(final_CSF, axis=0)[:, :, slice_idx], -1)
pixelwise_std_csf = np.expand_dims(np.std(final_CSF, axis=0)[:, :, slice_idx], -1)

pixelwise_mean_gm = np.expand_dims(np.mean(final_GM, axis=0)[:, :, slice_idx], -1)
pixelwise_std_gm = np.expand_dims(np.std(final_GM, axis=0)[:, :, slice_idx], -1)

pixelwise_mean_wm = np.expand_dims(np.mean(final_WM, axis=0)[:, :, slice_idx], -1)
pixelwise_std_wm = np.expand_dims(np.std(final_WM, axis=0)[:, :, slice_idx], -1)


for subject_idx in range(NUM_SUBJECTS):

    def perlin():
        # Maps all have noise added in the range [-10%, 10%] of their mean
        return generate_perlin_noise_3d((MAP_SIZE, MAP_SIZE, MAP_SIZE), (4, 4, 4))
    
    def gaussian():
        return np.random.randn((MAP_SIZE, MAP_SIZE, MAP_SIZE))
        

    def gen_map(
        gm_val, wm_val, csf_val, fat_val, csf_tissue, gm_tissue, wm_tissue
    ):
        #assert NO_NOISE, "perlin_noise not yet re-implemented"
        if NO_NOISE:
            total = (
                gm_tissue * gm_val +
                wm_tissue * wm_val +
                csf_tissue * csf_val
            )
        else:
            if NOISE_TYPE == 'perlin':
                total = (
                    gm_tissue * gm_val * (1 + 0.1 * perlin()) +
                    wm_tissue * wm_val * (1 + 0.1 * perlin()) +
                    csf_tissue * csf_val * (1 + 0.1 * perlin())
                )
            elif NOISE_TYPE =='gaussian':
                total = (
                    gm_tissue * gm_val * (1 + 0.1 * gaussian()) +
                    wm_tissue * wm_val * (1 + 0.1 * gaussian()) +
                    csf_tissue * csf_val * (1 + 0.1 * gaussian())
                )
        return total

    new_csf = np.random.normal(loc=pixelwise_mean_csf, scale=pixelwise_std_csf)
    draw(new_csf, os.path.join(csf_plot_dir, f"csf_{subject_idx}.png"))
    new_gm = np.random.normal(loc=pixelwise_mean_gm, scale=pixelwise_std_gm)
    new_wm = np.random.normal(loc=pixelwise_mean_wm, scale=pixelwise_std_wm)

    all_tensors = np.concatenate([
        new_csf.squeeze()[None, ...],
        new_gm.squeeze()[None, ...],
        new_wm.squeeze()[None, ...]]).swapaxes(0, 1).swapaxes(1, 2)
    
    not_bkg = np.array(
        ~(((all_tensors == 0)).sum(-1, keepdims=True)==3),
        dtype=np.int32
    )

    discrete = np.array((all_tensors.argmax(-1, keepdims=True)+1) * not_bkg, dtype=np.float32)

    T1_map = gen_map(1.55, 0.83, 4.16, 0.374, new_csf, new_gm, new_wm)
    T2_map = gen_map(0.09, 0.07, 1.65, 0.125, new_csf, new_gm, new_wm)
    # These are calculated from T2* (for which sources are sparse)
    T2dash_map = gen_map(0.322, 0.183, 0.0591, 0.0117, new_csf, new_gm, new_wm)
    # These are completeley guessed
    PD_map = gen_map(0.8, 0.7, 1.0, 1.0, new_csf, new_gm, new_wm)
    # Isometric diffusion in [10^-3 mm^2/s]
    D_map = gen_map(0.83, 0.65, 3.19, 0.1, new_csf, new_gm, new_wm)

    # Generate B0 and B1 maps for brainweb dataset
    B0, B1 = generate_b0_b1_map_for_brainweb(PD_map)
    B1 = B1.squeeze(0)

    # Save generated data
    name = f"subject{subject_idx:03d}.npz"
    file_name = os.path.join(output_dir, name)
    print(f"Saving maps to f'{output_dir}/subject{subject_idx:03d}.npz'")

    T1_map = torch.nn.functional.interpolate(
        torch.from_numpy(T1_map)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    T2_map = torch.nn.functional.interpolate(
        torch.from_numpy(T2_map)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    PD_map = torch.nn.functional.interpolate(
        torch.from_numpy(PD_map)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    T2dash_map = torch.nn.functional.interpolate(
        torch.from_numpy(T2dash_map)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    D_map = torch.nn.functional.interpolate(
        torch.from_numpy(D_map)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    new_wm = torch.nn.functional.interpolate(
        torch.from_numpy(new_wm)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    new_gm = torch.nn.functional.interpolate(
        torch.from_numpy(new_gm)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    new_csf = torch.nn.functional.interpolate(
        torch.from_numpy(new_csf)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    B0 = torch.nn.functional.interpolate(
        B0[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    B1 = torch.nn.functional.interpolate(
        B1[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1), 
    mode="area").squeeze().unsqueeze(-1).detach().numpy()
    discrete = torch.nn.functional.interpolate(
        torch.from_numpy(discrete)[None, None, ...],
        size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 1),
        mode="nearest"
    ).squeeze().unsqueeze(-1).detach().numpy()

    draw(new_csf, os.path.join(csf_plot_dir, f"csf_{subject_idx}.png"))
    draw(new_gm, os.path.join(gm_plot_dir, f"gm_{subject_idx}.png"))
    draw(new_wm, os.path.join(wm_plot_dir, f"wm_{subject_idx}.png"))
    draw(discrete, os.path.join(discrete_plot_dir, f"discrete_{subject_idx}.png"), clim=(0, 3))

    # data = torch.nn.functional.interpolate(torch.from_numpy(data)[None, None, ...],size=(64, 64, 432))

    if SAVE_MIDDLE_SLICE:
        num_slices = T1_map.shape[-1]
        T1_map = (T1_map[:, :, 0])[..., None]
        T2_map = (T2_map[:, :, 0])[..., None]
        T2dash_map = (T2dash_map[:, :, 0])[..., None]
        PD_map = (PD_map[:, :, 0])[..., None]
        D_map = (D_map[:, :, 0])[..., None]
        new_wm = (new_wm[:, :, 0])[..., None]
        new_gm = (new_gm[:, :, 0])[..., None]
        new_csf = (new_csf[:, :, 0])[..., None]
        discrete = (discrete[:, :, 0])[..., None]
        B0, B1 = generate_b0_b1_map_for_brainweb(PD_map)
        B1 = B1.squeeze(0)

        np.savez_compressed(
            file_name,
            T1_map=T1_map,
            T2_map=T2_map,
            T2dash_map=T2dash_map,
            PD_map=PD_map,
            D_map=D_map,
            tissue_WM = new_wm,
            tissue_GM = new_gm,
            tissue_CSF = new_csf,
            discrete = discrete,
            B0 = B0,
            B1 = B1
        )
