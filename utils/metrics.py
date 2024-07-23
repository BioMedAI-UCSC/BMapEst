import torch
import os
import argparse
import torch.nn as nn
from torchvision.models import vgg19
from torchmetrics import StructuralSimilarityIndexMeasure

def parse_args():
    parser = argparse.ArgumentParser(description="Metrics calculation script")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the metrics directory of an experiment")
    parser.add_argument("--gtdir", "-gt", type=str, default="ground_truth",
                        help="Directory name of ground truth prob maps tensors")
    parser.add_argument("--ssim", action="store_true", help="Reports SSIM")
    parser.add_argument("--l2", action="store_true", help="Reports L2 loss")
    parser.add_argument("--vgg", action="store_true", help="Report VGG loss")
    parser.add_argument("--psnr", action="store_true", help="Report PSNR")
    parser.add_argument("--specific_epochs", "-sp", action="store_true",
                        help="Enables metric calculation for specific epochs")
    parser.add_argument("--epochs", nargs="+",
                        help="Epoch numbers for metric calculation")

    args = parser.parse_args()
    return args

class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input.repeat([1, 3, 1, 1]))
        vgg_target_features = self.vgg(target.repeat([1, 3, 1, 1]))

        return self.loss(vgg_input_features, vgg_target_features)

class Metrics():
    def __init__(
        self,
        metrics_dir: str,
        gt_dir: str,
        ssim: bool = True,
        l2: bool = True,
        vgg: bool = True,
        psnr: bool = True,
        device: str = "cpu"
    ):
        self.metrics_dir = metrics_dir
        self.gt_dir = gt_dir
        self.ssim = ssim
        self.l2 = l2
        self.vgg = vgg
        self.psnr = psnr
        self.device = device
        self.metrics_list = []
        self._setup_metrics()
    
    def calculate_psnr(self, pred, target):
        mse = torch.mean((pred-target)**2)
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel/torch.sqrt(mse))
        return psnr

    def _setup_metrics(self):
        self.heading = "tissue"
        if self.l2:
            self.heading += ",l2"
            self.l2_metric = lambda inp, target: torch.mean((inp-target)**2)
        if self.ssim:
            self.heading += ",ssim"
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        if self.vgg:
            self.heading += ",vgg"
            self.vgg_metric = PerceptualLoss(self.device)
        if self.psnr:
            self.heading += ",psnr"
            self.psnr_metric = lambda pred, tgt: self.calculate_psnr(
                pred, tgt
            )
        self.heading += "\n"

    def get_metrics(self, input_tensor, target_tensor, tissue_name="UNK"):
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input tensor is not a PyTorch Tensor")
        if not isinstance(target_tensor, torch.Tensor):
            raise TypeError("Target tensor is not a PyTorch Tensor")
        string = f"{tissue_name}"
        if self.l2:
            metric = self.l2_metric(input_tensor, target_tensor)
            string += f",{metric}"
        if self.ssim:
            temp_input_tensor = input_tensor.squeeze()[None, None, ...]
            temp_target_tensor = target_tensor.squeeze()[None, None, ...]
            metric = self.ssim_metric(temp_input_tensor, temp_target_tensor)
            string += f",{metric}"
        if self.vgg:
            temp_input_tensor = input_tensor.squeeze()[None, None, ...]
            temp_target_tensor = target_tensor.squeeze()[None, None, ...]
            temp_input_tensor = temp_input_tensor.to(self.device)
            temp_target_tensor = temp_target_tensor.to(self.device)
            metric = self.vgg_metric(temp_input_tensor, temp_target_tensor)
            string += f",{metric}"
        if self.psnr:
            metric = self.psnr_metric(input_tensor, target_tensor)
            string += f",{metric}"
        return string + "\n"

    def update_metrics_list(self, metrics):
        self.metrics_list.append(metrics)
    
    def flush_metrics_list(self, path):
        with open(path, 'w') as writer:
            writer.write(self.heading)
            for metrics in self.metrics_list:
                writer.write(metrics)
        self.metrics_list = []
    
    def generate_metrics(self):
        csf_gt_path = os.path.join(self.gt_dir, "csf.tensor")
        optim_csf = optim_gm = optim_wm = False
        if os.path.exists(csf_gt_path):
            optim_csf = True
            gt_csf = torch.load(
                csf_gt_path,
                map_location=self.device
            ).detach()
        gm_gt_path = os.path.join(self.gt_dir, "gm.tensor")
        if os.path.exists(gm_gt_path):
            gt_gm = torch.load(
                gm_gt_path,
                map_location=self.device
            ).detach()
            optim_gm = True
        wm_gt_path = os.path.join(self.gt_dir, "wm.tensor")
        if os.path.exists(wm_gt_path):
            gt_wm = torch.load(
                wm_gt_path,
                map_location=self.device
            ).detach()
            optim_wm = True

        all_epochs = [
            os.path.join(args.input, dir) for dir in os.listdir(args.input)
            if dir.startswith("epoch_")
        ]

        if args.specific_epochs:
            temp = []
            for epoch in all_epochs:
                epoch_num = os.path.split(epoch)[1].split("_")[1]
                if epoch_num in args.epochs:
                    temp.append(epoch)
            all_epochs = temp
        
        for epoch in all_epochs:
            epoch_string = os.path.split(epoch)[1]
            print(f"Generating metrics for: {epoch_string}")
            writer = open(
                os.path.join(args.input, epoch_string, "metrics.csv"), "w"
            )
            writer.write(self.heading)
            with torch.no_grad():
                csf_tensor_path = os.path.join(epoch, "csf.tensor")
                csf_metrics = ""
                if optim_csf:
                    epoch_csf = torch.load(
                        csf_tensor_path,
                        map_location="cpu"
                    )
                    csf_metrics = self.get_metrics(epoch_csf, gt_csf, "CSF")
                
                gm_tensor_path = os.path.join(epoch, "gm.tensor")
                gm_metrics = ""
                if optim_gm:
                    epoch_gm = torch.load(
                        gm_tensor_path,
                        map_location="cpu"
                    )
                    gm_metrics = self.get_metrics(epoch_gm, gt_gm, "GM")

                wm_tensor_path = os.path.join(epoch, "wm.tensor")
                wm_metrics = ""
                if optim_wm:
                    epoch_wm = torch.load(
                        wm_tensor_path,
                        map_location="cpu"
                    )
                    wm_metrics = self.get_metrics(epoch_wm, gt_wm, "WM")
                writer.write(csf_metrics + gm_metrics + wm_metrics)
            writer.close()

if __name__ == "__main__":
    args = parse_args()
    args.input = os.path.join(args.input, "metrics")
    ground_truths = os.path.join(args.input, "ground_truths")
    metrics = Metrics(
        metrics_dir=args.input, gt_dir=ground_truths, ssim=args.ssim,
        l2=args.l2, vgg=args.vgg, psnr=args.psnr
    )
    metrics.generate_metrics()