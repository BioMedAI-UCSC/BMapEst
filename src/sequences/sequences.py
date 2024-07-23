import MRzeroCore as mr0
import torch
import os

from numpy import pi
from utils.util import draw, draw_single_image

class Sequences():
    def __init__(self, args, logger):
        self.Nread = args.nread
        self.Nphase = args.nphase
        self.seq = None
        self.logger = logger
        self.args = args
        self.target = None
        self.obj_p = None
        self.sequence_type = "UNDEFINED"
        self.loss_history = []
        self.phantom = None
        self.fov = torch.ones(3)

    def build(self):
        """
            This method builds the sequence using PyPulseq library.
        """
        raise NotImplementedError()

    def check_timings(self):
        """
            This method checks the timings of the sequences.
        """
        if isinstance(self.seq, type(None)):
            raise RuntimeError("Sequence is not yet build!")

        ok, error_report = self.seq.check_timing()
        if ok:
            print('Timing check passed successfully')
        else:
            print('Timing check failed. Error listing follows:')
            for e in error_report:
                print(e)

    def setup_ground_truth(self):
        """
            Method that sets up the ground truth for optimization. The sequence 
            is simulated on the obj_p object. Based on the loss function the 
            target variable is set accordingly. This method must be implemented 
            in the child class.
            Parameter
            ---------
            Return
            ------
                None
        """
        raise NotImplementedError()

    def run_mrzero_pipeline(self):
        """
            Method to execute the MRTwin simulator on the built obj_p object.
            Paramter. The version of MRzeroCore==0.2.9
            --------
            Return
            ------
                signal: torch.complex64
                    Signal from the MRI simulator
        """
        graph = mr0.compute_graph(self.seq, self.obj_p, 200, 1e-3)
        signal = mr0.execute_graph(graph, self.seq, self.obj_p)
        return signal

    def save_plots(
            self, plots_dir, pred, target, start_idx=0,
            plot_kspace_difference=False
    ):
        """
            Method to save the ground truth and optimized probability maps.
            Parameter
            ---------
                epoch: int
                    Current epoch number
                pred: torch.Tensor
                    Output tensor with which the loss is calculated
            Return
            ------
                None
        """

        for i in range(pred.shape[0]):
            save_at = os.path.join(
                plots_dir, f"{self.sequence_type}_space_{start_idx+i}.png"
            )
            tgt = target[i].detach().cpu().numpy()
            pd = pred[i].detach().cpu().numpy()

            #TODO: Adding this line of code for debugging
            if plot_kspace_difference:
                new = abs(tgt - pd)
                save_filename = \
                    f"{self.sequence_type}_kspace_diff_{start_idx+i}.png"
                draw_single_image(
                    new, clim=[0, 5], image_name=f"kspace_diff_{i}",
                    save_at=os.path.join(plots_dir, save_filename)
                )

            draw(tgt, pd, gt_name=f"target_{start_idx+i}", 
                 pred_name=f"pred_{start_idx+i}", save_at=save_at, clim=None)

    def forward_pass(self, build_gt=False):
        """
            Method to perform the MRI simulation using the MRTwin simulator.
            This method must be implemted in the child class.
            Parameter
            ---------
                build_gt: bool
                    Identifier to build the ground truth instead of performing
                    the optimization
            Return
            ------
                torch.Tensor
                    Output Tensor with which the loss is computed for
                    backpropagation.
        """
        raise NotImplementedError()

    def calc_avg_B1_trig(self, B1: torch.Tensor, PD: torch.Tensor) -> torch.Tensor:
        """Return a (361, 3) tensor for B1 specific sin, cos and sin² values.

        This function calculates values for sin, cos and sin² for (0, 2pi) * B1 and
        then averages the results, weighted by PD. These 3 functions are the non
        linear parts of a rotation matrix, the resulting look up table can be used
        to calcualte averaged rotations for the whole phantom. This is useful for
        the pre-pass, to get better magnetization estmates even if the pre-pass is
        not spatially resolved.
        """
        # With pTx, there are now potentially multiple B1 maps with phase.
        # NOTE: This is a (probably suboptimal) workaround
        B1 = B1.sum(0).abs()

        B1 = B1.flatten()[:, None]  # voxels, 1
        PD = (PD.flatten() / PD.sum())[:, None]  # voxels, 1
        angle = torch.linspace(0, 2*pi, 361, device=PD.device)[None, :]  # 1, angle
        return torch.stack([
            (torch.sin(B1 * angle) * PD).sum(0),
            (torch.cos(B1 * angle) * PD).sum(0),
            (torch.sin(B1 * angle/2)**2 * PD).sum(0)
        ], dim=1).type(torch.float32)