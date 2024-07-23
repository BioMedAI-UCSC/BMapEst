import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="PSNR calculator")
    parser.add_argument("--subject", "-s", type=str, required=True,
                        help="Path to the subject directory")
    parser.add_argument("--epoch", type=int, default=500,
                        help="Epoch Number")
    parser.add_argument("--metrics", "-m", type=str, default="metrics",
                        help="Name of the metrics directory")
    parser.add_argument("--dice", "-d", action="store_true")
    parser.add_argument("--crisp_gtdir", "-c", type=str, default=None,
                        help="Loads the discrete model for Dice evaluation")
    return parser.parse_args()

def PSNR(gt, pred):
    mse = torch.mean((gt-pred)**2)
    max_pixel = 1
    psnr = 20 * torch.log10(max_pixel/torch.sqrt(mse))
    return psnr

def DICE(gt: torch.Tensor, pred: torch.Tensor):
    return ((2. * gt.logical_and(pred))).sum() / (gt.sum() + pred.sum() + 1e-6)

def main(args):
    experiments = [os.path.join(args.subject, dir) for dir in os.listdir(args.subject)]
    subject_filename = os.path.split(args.subject)[1]
    gt_crisp = np.load(os.path.join(args.crisp_gtdir, subject_filename))['discrete'].squeeze()
    # gt_crisp_bck = torch.from_numpy(np.array(gt_crisp == 0, dtype=np.int32))
    gt_crisp_csf = torch.from_numpy(np.array(gt_crisp == 1, dtype=np.int32))
    gt_crisp_gm = torch.from_numpy(np.array(gt_crisp == 2, dtype=np.int32))
    gt_crisp_wm = torch.from_numpy(np.array(gt_crisp == 3, dtype=np.int32))

    for exp in experiments:
        optim_csf = optim_gm = optim_wm = False
        metrics_dir = os.path.join(exp, args.metrics)
        gt_dir = os.path.join(metrics_dir, "ground_truths")
        csf_gt = gm_gt = wm_gt = None
        gt_stacked = None
        classes = {
            "csf" : None,
            "gm" : None,
            "wm" : None
        }
        pred_stacked = None
        epoch_dir = os.path.join(metrics_dir, f"epoch_{args.epoch}")
        psnr_dir = os.path.join(epoch_dir, "psnr_dir")
        dice_dir = os.path.join(epoch_dir, "dice_dir")
        os.makedirs(dice_dir, exist_ok=True)
        os.makedirs(psnr_dir, exist_ok=True)

        for idx, tensor in enumerate(os.listdir(gt_dir)):
            if tensor.split(".")[0]=="csf":
                csf_gt = torch.load(os.path.join(gt_dir, tensor))
                optim_csf = True
                csf_pred = torch.load(os.path.join(epoch_dir, "csf.tensor"))
                with open(os.path.join(psnr_dir, "csf_psnr.txt"), "w") as writer:
                    psnr = PSNR(csf_gt, csf_pred)
                    writer.write(f"{round(psnr.item(), 3)}")
                if isinstance(gt_stacked, type(None)):
                    gt_stacked = torch.unsqueeze(csf_gt.squeeze(), 0)
                    pred_stacked = torch.unsqueeze(csf_pred.squeeze(), 0)
                else:
                    gt_stacked = torch.concatenate([gt_stacked, csf_gt.squeeze().unsqueeze(0)])
                    pred_stacked = torch.concatenate([pred_stacked, csf_pred.squeeze().unsqueeze(0)])
                classes["csf"] = idx
            elif tensor.split(".")[0]=="gm":
                gm_gt = torch.load(os.path.join(gt_dir, tensor))
                optim_gm = True
                gm_pred = torch.load(os.path.join(epoch_dir, "gm.tensor"))
                with open(os.path.join(psnr_dir, "gm_psnr.txt"), "w") as writer:
                    psnr = PSNR(gm_gt, gm_pred)
                    writer.write(f"{round(psnr.item(), 3)}")
                if isinstance(gt_stacked, type(None)):
                    gt_stacked = torch.unsqueeze(gm_gt.squeeze(), 0)
                    pred_stacked = torch.unsqueeze(gm_pred.squeeze(), 0)
                else:
                    gt_stacked = torch.concatenate([gt_stacked, gm_gt.squeeze().unsqueeze(0)])
                    pred_stacked = torch.concatenate([pred_stacked, gm_pred.squeeze().unsqueeze(0)])
                classes["gm"] = idx
            elif tensor.split(".")[0]=="wm":
                wm_gt = torch.load(os.path.join(gt_dir, tensor))
                optim_wm = True
                wm_pred = torch.load(os.path.join(epoch_dir, "wm.tensor"))
                with open(os.path.join(psnr_dir, "wm_psnr.txt"), "w") as writer:
                    psnr = PSNR(wm_gt, wm_pred)
                    writer.write(f"{round(psnr.item(), 3)}")
                if isinstance(gt_stacked, type(None)):
                    gt_stacked = torch.unsqueeze(wm_gt.squeeze(), 0)
                    pred_stacked = torch.unsqueeze(wm_pred.squeeze(), 0)
                else:
                    gt_stacked = torch.concatenate([gt_stacked, wm_gt.squeeze().unsqueeze(0)])
                    pred_stacked = torch.concatenate([pred_stacked, wm_pred.squeeze().unsqueeze(0)])
                classes["wm"] = idx

        # gt_class = torch.argmax(gt_stacked, 0)
        # import pdb; pdb.set_trace()

        # pred_stacked: (3, 64, 64)
        # pred_class: (64, 64) : unique(0, 1, 2)
        # pred_bkg: (3, 64, 64)
        # num_classes = 3
        # not_pred_bkg = (64, 64)
        # pred_bkg = (64, 64)
        # classes = {'csf': 1, 'gm' : 0, 'wm' : 2}

        # if os.path.split(exp)[1] == "exp_direct_pixel_wm_only":
        #     import pdb; pdb.set_trace()
        pred_class = torch.argmax(pred_stacked, 0)
        pred_bkg = pred_stacked.clone()
        pred_bkg[pred_bkg < 0.35] = 0
        num_classes = int(optim_csf) + int(optim_gm) + int(optim_wm)
        not_pred_bkg = (~((pred_bkg==0).sum(0)==num_classes)).type(torch.int32)
        pred_bkg = ((pred_bkg==0).sum(0)==num_classes).type(torch.int32)

        csf_dice = gm_dice = wm_dice = None
        csf_pred_for_dice = gm_pred_for_dice = wm_pred_for_dice = None
        if optim_csf:
            csf_pred_for_dice = (
                (pred_class == classes["csf"]) * not_pred_bkg
            ).type(torch.int32)
            with open(os.path.join(dice_dir, "csf_dice.txt"), "w") as writer:
                csf_dice = DICE(gt_crisp_csf, csf_pred_for_dice)
                writer.write(f"{round(csf_dice.item(), 3)}")

        if optim_gm:
            gm_pred_for_dice = (
                (pred_class == classes["gm"]) * not_pred_bkg
            ).type(torch.int32)
            with open(os.path.join(dice_dir, "gm_dice.txt"), "w") as writer:
                gm_dice = DICE(gt_crisp_gm, gm_pred_for_dice)
                writer.write(f"{round(gm_dice.item(), 3)}")

        if optim_wm:
            wm_pred_for_dice = (
                (pred_class == classes["wm"]) * not_pred_bkg
            ).type(torch.int32)
            with open(os.path.join(dice_dir, "wm_dice.txt"), "w") as writer:
                wm_dice = DICE(gt_crisp_wm, wm_pred_for_dice)
                writer.write(f"{round(wm_dice.item(), 3)}")
                

        print(f"{os.path.split(args.subject)[1]:<15}: {os.path.split(exp)[1]:>40} : \
              {round(csf_dice.item(), 2) if not isinstance(csf_dice, type(None)) else None}, \
              {round(gm_dice.item(), 2) if not isinstance(gm_dice, type(None)) else None}, \
              {round(wm_dice.item(), 2) if not isinstance(wm_dice, type(None)) else None}"
        )
    print("-"*150)

if __name__ == "__main__":
    main(parse_args())