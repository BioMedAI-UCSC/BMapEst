import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.archs.unet import UNet
from src.dataset import ProbmapsAndScansDataset
from torch.utils.data import Dataset, DataLoader
from utils.util import PSNR, DICE
from torchmetrics import StructuralSimilarityIndexMeasure
torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline training script")
    parser.add_argument("--exp", "-e", type=str, required=True,
                        help="Experiment ID")
    parser.add_argument("--data", "-d", type=str, default="data",
                        help="Path to the data directory")
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="Path to store the results")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Places simulation either on CPU or GPU")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs")
    parser.add_argument("--optimize_csf", action="store_true",
                        help="Flag to optimize CSF")
    parser.add_argument("--optimize_wm", action="store_true",
                        help="Flag to optimize White Matter")
    parser.add_argument("--optimize_gm", action="store_true",
                        help="Flag to optimize GM")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=['adam', 'adamw', 'sgd'],
                        help="Supported optimizers")
    parser.add_argument("--losses", type=str, default="mse", nargs='+',
                        help="Loss types")
    parser.add_argument("--num_sequences", type=int, default=6,
                        help="Number of sequences to use")
    parser.add_argument("--arch", type=str, default="unet",
                        choices=["unet"],
                        help="Neural Network to train")
    parser.add_argument("--batch", type=int, default=32,
                           help="Batch size to train the neural network")
    parser.add_argument("--num_workers", type=int, default=4,
                           help="Number of workers for DataLoader")
    args = parser.parse_args()

    if args.optimize_csf==args.optimize_gm==args.optimize_wm==False:
        args.optimize_csf = args.optimize_gm = args.optimize_wm = True
    
    args.exp = f"exp_{args.exp}"
    args.output = os.path.join(args.output, args.exp)
    args.plots_dir = os.path.join(args.output, "plots")
    args.metrics = os.path.join(args.output, "metrics")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.metrics, exist_ok=True)
    
    return args

def setup_dataset(args):
    train_dataset = ProbmapsAndScansDataset(args, args.data)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, drop_last=True,
        shuffle=True
    )
    test_dataset = ProbmapsAndScansDataset(args, args.data, is_train=False)
    val_loader = DataLoader(
        test_dataset, batch_size=args.batch, drop_last=False,
        shuffle=False
    )

    return train_dataset, train_loader, test_dataset, val_loader

def main(args):
    train_dataset, train_loader, test_dataset, test_loader = setup_dataset(args)
    loaders = {"train": train_loader, "test": test_loader}

    optimizing_maps = [
        ['CSF', args.optimize_csf],
        ['GM' , args.optimize_gm],
        ['WM' , args.optimize_wm]
    ]
    num_output_channels = args.optimize_csf + args.optimize_gm + \
                                                                args.optimize_wm
    SSIM = StructuralSimilarityIndexMeasure(
        data_range=1.0, reduction=None
    ).to(args.device)
    
    # in_channels = 4 because one for sequence idx and four for the 4 contrasts
    # out_channels will depend upon the number of prob map estimation

    model = UNet(
        in_channels=4*args.num_sequences if args.num_sequences!=0 else 1,
        out_channels=num_output_channels
    ).to(args.device)

    if args.num_sequences==0:
        args.num_sequences = 1

    optimizer = optim.Adam(model.parameters())
    best_psnr = -float("inf")

    metrics_writer = open(os.path.join(args.output, "metrics.txt"), "w")

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        dice_epoch = {}
        ssim_epoch = {}
        psnr_epoch = {}
        loss_train = []
        loss_test = []
        mean_psnr = 0
        mean_ssim = 0
        mean_dice = 0
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for idx, data in enumerate(loaders[phase]):
                input_image_space, target_phantom, discrete, filenames = data
                input_image_space = input_image_space.to(args.device)
                target_phantom = target_phantom.to(args.device)
                discrete = discrete.to(args.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # predicted_phantom {batch_size, num_output_channels, height, width}
                    predicted_phantom = model(input_image_space)
                    
                    predicted_not_bkg_mask = (
                            ~(((predicted_phantom < 0.01).sum(1, keepdim=True))
                        ==num_output_channels)
                    ).type(torch.int32)

                    # The ordering is CSF, GM, and WM respectively
                    classes = (torch.argmax(
                        predicted_phantom, dim=1, keepdim=True
                    )+1) * predicted_not_bkg_mask

                    idx = 0
                    dice_loss = 0
                    num_samples = 0
                    if phase=="test":
                        for probmap, flag in optimizing_maps:
                            if flag:
                                pred = (classes[:, 0, :, :] == (idx+1))[:, None, ...]
                                gt = discrete[:, idx, :, :][:, None, ...]
                                if epoch%1==0:
                                    epoch_plot_dir = os.path.join(args.plots_dir, f"epoch_{epoch}")
                                    for subject_idx, filepath in enumerate(filenames):
                                        dir_name = (os.path.split(filepath)[1]).replace(".npz", "")
                                        plot_dir = os.path.join(epoch_plot_dir, dir_name)
                                        os.makedirs(plot_dir, exist_ok=True)
                                        map_ = predicted_phantom[subject_idx, idx, :, :].clone().detach().cpu().numpy()
                                        figure = plt.figure(figsize=(8, 8))
                                        plt.imshow(map_, clim=(0, 1))
                                        plt.axis('off')
                                        plt.savefig(os.path.join(plot_dir, f"{probmap}_{dir_name}.png"), bbox_inches='tight', pad_inches=0)
                                        plt.close(figure)



                                dice_score = DICE(gt, pred, reduction=False)
                                num_samples += gt.shape[0]
                                dice_loss = dice_loss + dice_score
                                psnr = PSNR(
                                    target_phantom[:, idx, :, :].unsqueeze(1),
                                    predicted_phantom[:, idx, :, :].unsqueeze(1),
                                    reduction=False
                                )
                                ssim = SSIM(
                                    target_phantom[:, idx, :, :].unsqueeze(1),
                                    predicted_phantom[:, idx, :, :].unsqueeze(1)
                                )
                                if probmap not in dice_epoch:
                                    dice_epoch[probmap] = dice_score.cpu().numpy().tolist()
                                    psnr_epoch[probmap] = psnr.cpu().numpy().tolist()
                                    ssim_epoch[probmap] = ssim.cpu().numpy().tolist()
                                else:
                                    dice_epoch[probmap] += dice_score.cpu().numpy().tolist()
                                    psnr_epoch[probmap] += psnr.cpu().numpy().tolist()
                                    ssim_epoch[probmap] += ssim.cpu().numpy().tolist()
                                idx += 1

                    # print(torch.mean((target_phantom - predicted_phantom) ** 2))
                    loss = torch.mean((target_phantom - predicted_phantom) ** 2)# + (1-(dice_loss/num_samples))

                    if phase == "test":
                        loss_test.append(loss.item())

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
            if phase == "train":
                training_loss = round(sum(loss_train) / len(loss_train), 5)
                print(f"\nEpoch: {epoch}: Training Loss: {training_loss}\n")

            if phase == "test":
                testing_loss = round(sum(loss_test) / len(loss_test), 3)
                print(f"\nEpoch: {epoch}: Testing Loss: {testing_loss}\n")
                metrics_writer.write(
                    f"Epoch: {epoch}\n---------------------------\n"
                )
                for probmap, flag in optimizing_maps:
                    if not flag:
                        continue
                    psnr_list = np.array(psnr_epoch[probmap])
                    ssim_list = np.array(ssim_epoch[probmap])
                    dice_list = np.array(dice_epoch[probmap])
                    mean_psnr += np.mean(psnr_list)
                    mean_ssim += np.mean(ssim_list)
                    mean_dice += np.mean(dice_list)
                    print(f"{probmap:<4}: Dice: {round(np.mean(dice_list), 2):>6}+-{round(np.std(dice_list), 3):<6}, PSNR: {round(np.mean(psnr_list), 2):>6}+-{round(np.std(psnr_list), 3):<6}, SSIM: {round(np.mean(ssim_list), 2):>6}+-{round(np.std(ssim_list), 3):<6}")
                    metrics_writer.write(f"{probmap:<4} PSNR: {round(np.mean(psnr_list), 3):>6}+-{round(np.std(psnr_list), 3):<6}\n")
                    metrics_writer.write(f"{probmap:<4} SSIM: {round(np.mean(ssim_list), 3):>6}+-{round(np.std(ssim_list), 3):<6}\n")
                    metrics_writer.write(f"{probmap:<4} DICE: {round(np.mean(dice_list), 3):>6}+-{round(np.std(dice_list), 3):<6}\n")
                metrics_writer.write("---------------------------\n")
                metrics_writer.flush()
                mean_psnr = mean_psnr / num_output_channels
                mean_ssim = mean_ssim / num_output_channels
                mean_dice = mean_dice / num_output_channels

            
            if phase == "test" and mean_psnr > best_psnr:
                state_dict = model.state_dict()
                best_psnr = mean_psnr
                torch.save(
                    {
                        "psnr" : round(best_psnr, 2),
                        "ssim" : round(mean_ssim, 2),
                        "dice" : round(mean_dice, 2),
                        "state_dict" : state_dict,
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "epoch" : epoch
                    }, os.path.join(args.output, "model_best.pt")
                )
                print(f"Saved model for epoch {epoch} at best PSNR: {round(best_psnr, 2)}")

if __name__=="__main__":
    main(parse_args())