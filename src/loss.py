import torch
import os
import matplotlib.pyplot as plt

from torchmetrics import StructuralSimilarityIndexMeasure

class Loss():
    def __init__(self, args, logger):
        """Implements different types of losses for the output"""
        self.args = args
        self.logger = logger
        self.loss_history = []
        self._init_losses()
    
    def _init_losses(self):
        for loss in self.args.losses:
            if loss == "ssim":
                self.logger.info("Initializing SSIM loss!")
                self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
                self.ssim_lambda = 10
    
    def calc_loss(self, targets, preds):
        """
            This method calculates the loss between the targets and predictions.
            Parameter
            ---------
                targets: List[torch.Tensor]
                    Target tensors
                preds: List[torch.Tensor]
                    Predicted tensors
            Return
            ------
                Tuple(torch.Tensor, torch.Tensor)
                    Loss and output tensor that was used to calculate the loss.
        """

        loss_sum = 0
        for id, target, pred in zip(list(range(0, len(preds))), targets, preds):
            for loss in self.args.losses:
                if loss == "mse":
                    mse_loss = torch.mean((target-pred)**2)
                    if id%2!=0 and self.args.kspace_smoothing:
                        mse_loss = 2.19 * mse_loss
                    loss_sum = loss_sum + mse_loss
                if loss == "ssim":
                    ssim_loss = self.ssim(
                        torch.unsqueeze(target, 0),
                        torch.unsqueeze(pred, 0)
                    ) * self.ssim_lambda
                    loss_sum = loss_sum + ssim_loss
        self.loss_history.append(loss_sum.detach().cpu().numpy().round(1))
        return loss_sum
    
    def save_plots(self, plots_dir):
        fig = plt.figure(figsize=(16, 16))
        plt.plot(list(range(0, len(self.loss_history))), self.loss_history)
        plt.savefig(os.path.join(plots_dir, "loss.png"))
        plt.close(fig)