import time
import torch
import os
import sys

from utils.util import draw
from .sequences.flash_mutliple_inversion import FlashMultipleInversion
from .sequences.mc_flash import McFlash
from .brain_probability_maps import BrainProbabilityMaps
from .optimizers import Optimizers
from .loss import Loss
from .upsample import Upsample

class Pipeline():
    def __init__(self, args, logger):
        """
        Pipeline for optimizing the probability maps using multiple sequences.
        """
        self.args = args
        self.logger = logger
        self.sigma = self.args.sigma
        self.sequences = []

    def build_pipeline(self):
        # Building the sequeneces objects
        for sequence_name in self.args.sequences:
            if sequence_name.lower() == "flash_multiple_inversion":
                seq_obj = FlashMultipleInversion(self.args, self.logger)
            elif sequence_name.lower() == "mc_flash":
                seq_objs_list = []
                for seq_idx, seq_type in enumerate(McFlash.seq_types):
                    seq_obj = McFlash(
                        self.args, self.logger, sequence_type=seq_type,
                        num_contrasts=1 if self.args.flash_seq_num==0 else 4
                    )
                    if self.args.generate_training_data:
                        seq_obj.generate_training_data()
                        self.logger.info("Training data generation is over!")
                        continue

                    seq_obj.setup_ground_truth()
                    seq_obj.phantom.B0 = seq_obj.phantom.B0.to(self.args.device)
                    seq_obj.phantom.B1 = seq_obj.phantom.B1.to(self.args.device)
                    seq_objs_list.append(seq_obj)
                    if (seq_idx+1) == self.args.flash_seq_num or \
                            self.args.flash_seq_num==0:
                        break
            else:
                raise NotImplementedError(sequence_name)
            
            if not isinstance(seq_obj, list):
                self.sequences = self.sequences + seq_objs_list
            else:
                seq_obj.setup_ground_truth()
                seq_obj.phantom.B0 = seq_obj.phantom.B0.to(self.args.device)
                seq_obj.phantom.B1 = seq_obj.phantom.B1.to(self.args.device)
                self.sequences.append(seq_obj)
            self.logger.info(f"{sequence_name}: Ground truth setup is done!")

        if self.args.generate_training_data:
            # Exit because only training data generation was required
            sys.exit(0)

        # Build the probability maps
        self.setup_probmaps()
        self.logger.info("Input probability maps or coef setup done!")

        self.optimizers = Optimizers(self.args, self.logger, self.probmaps)
        self.logger.info("PyTorch Optimizers setup is done!")

        if self.args.upsample:
            self.csf_upsample = Upsample(
                in_chans=1, out_chans=1, tissue_name="CSF",
                device=self.args.device
            )
            self.csf_upsample.groud_truth = self.probmaps.upsampled_tissueCSF
            self.gm_upsample = Upsample(
                in_chans=1, out_chans=1, tissue_name="GM",
                device=self.args.device
            )
            self.gm_upsample.groud_truth = self.probmaps.upsampled_tissueGM
            self.wm_upsample = Upsample(
                in_chans=1, out_chans=1, tissue_name="WM",
                device=self.args.device
            )
            self.wm_upsample.groud_truth = self.probmaps.upsampled_tissueWM

        self.loss = Loss(self.args, self.logger)
        self.logger.info("Loss setup is done!")
    
    def setup_probmaps(self):
        """
            Method to setup the probability maps object. This object houses the 
            ground truth CSF, GM, and WM objects. It also builts the linear 
            coefficients for the probability map's coefficient optimization.
            Parameter
            ---------
            Return
            ------
                None
        """
        if self.args.upsample:
            assert self.args.nread%2==0, "Nread must be a multiple of 2!"
            assert self.args.nphase%2==0, "Nphase must be a multiple of 2!"
            probmap_shape = [self.args.nread//2, self.args.nphase//2, 1]
        else:
            probmap_shape = [self.args.nread, self.args.nphase, 1]

        self.probmaps = BrainProbabilityMaps(
            ground_truth_path=self.args.ground_truth,
            full_data_dir=self.args.data,
            upsampled_data_dir=self.args.upsampled_data,
            metrics_dir=self.args.metrics,
            ignore_subjects_list=self.args.ignore_sub,
            probmap_shape=probmap_shape,
            init_method=self.args.init_method,
            init_constant=self.args.init_constant,
            interpolated_idx=self.args.interpolated_idx,
            init_avg=self.args.init_avg,
            init_noise_to_add=self.args.init_noise,
            optim_csf=self.args.optimize_csf,
            optim_gm=self.args.optimize_gm,
            optim_wm=self.args.optimize_wm,
            pca_coeff_optim=self.args.pca_coeff_optim,
            num_pca_coefficients=self.args.num_pca_coeff,
            linear_combination_optim=self.args.linear_comb_optim,
            pixelwise_constant=self.args.pixelwise,
            coeff_multiplier=self.args.multiplier,
            logger=self.logger,
            device=self.args.device
        )

    def optimize_maps(self):
        """
            Entry method for running the forward MRI simulation. This has a loop
            which runs for a fixed number of epochs to optimize the probability 
            maps in case of normal map optimization and coefficients in the 
            case of linear coefficients optimization. This also saves the plots 
            of optimized and ground truth probability maps along with the 
            output tensors.
            Parameter
            ---------
            Return
            ------
                None
        """
        self.logger.info("---------------------")
        self.logger.info("Starting Optimization")
        self.logger.info("---------------------")

        num_epochs = self.args.epochs * sum(
            [self.args.optimize_csf, self.args.optimize_gm, 
             self.args.optimize_wm]
            ) if self.args.alternate_optim else self.args.epochs
        self.logger.info(f"Optimizing for {num_epochs} epochs!")
        sigmoid = torch.nn.Sigmoid()

        for epoch in range(num_epochs):
            tic = time.time()
            torch.cuda.empty_cache()
            # Write the code for PCA
            if self.args.linear_comb_optim or self.args.pca_coeff_optim:
                # Build intermediate tensors
                if self.args.optimize_csf:
                    inter_csf = torch.unsqueeze(
                        (self.probmaps.csf*self.probmaps.csf_coef).sum(axis=0),
                        axis=-1
                    )
                    if self.args.pca_coeff_optim:
                        inter_csf = sigmoid(inter_csf)
                    else:
                        inter_csf = torch.clamp(inter_csf, 0.0, 1.0)
                else:
                    inter_csf = self.probmaps.csf

                if self.args.optimize_gm:
                    inter_gm = torch.unsqueeze(
                        (self.probmaps.gm*self.probmaps.gm_coef).sum(axis=0),
                        axis=-1
                    )
                    if self.args.pca_coeff_optim:
                        inter_gm = sigmoid(inter_gm)
                    else:
                        inter_gm = torch.clamp(inter_gm, 0.0, 1.0)
                else:
                    inter_gm = self.probmaps.gm

                if self.args.optimize_wm:
                    inter_wm = torch.unsqueeze(
                        (self.probmaps.wm*self.probmaps.wm_coef).sum(axis=0),
                        axis=-1
                    )
                    if self.args.pca_coeff_optim:
                        inter_wm = sigmoid(inter_wm)
                    else:
                        inter_wm = torch.clamp(inter_wm, 0.0, 1.0)
                else:
                    inter_wm = self.probmaps.wm
            else:
                inter_csf = self.probmaps.csf
                inter_gm = self.probmaps.gm
                inter_wm = self.probmaps.wm
            
            if self.args.upsample:
                inter_csf = torch.clamp(
                    self.csf_upsample(
                        inter_csf.squeeze().unsqueeze(0).unsqueeze(0)
                    ), 0.0, 1.0).squeeze().unsqueeze(-1)
                inter_gm = torch.clamp(
                    self.gm_upsample(
                        inter_gm.squeeze().unsqueeze(0).unsqueeze(0)
                    ), 0.0, 1.0).squeeze().unsqueeze(-1)
                inter_wm = torch.clamp(
                    self.wm_upsample(
                        inter_wm.squeeze().unsqueeze(0).unsqueeze(0)
                    ), 0.0, 1.0).squeeze().unsqueeze(-1)

            pd = inter_gm * self.probmaps.timings['pd'][0] + \
                    inter_wm * self.probmaps.timings['pd'][1] + \
                    inter_csf * self.probmaps.timings['pd'][2]
            t1 = inter_gm * self.probmaps.timings['t1'][0] + \
                    inter_wm * self.probmaps.timings['t1'][1] + \
                    inter_csf * self.probmaps.timings['t1'][2]
            t2 = inter_gm * self.probmaps.timings['t2'][0] + \
                    inter_wm * self.probmaps.timings['t2'][1] + \
                    inter_csf * self.probmaps.timings['t2'][2]
            t2dash = inter_gm * self.probmaps.timings['t2_dash'][0] + \
                        inter_wm * self.probmaps.timings['t2_dash'][1] + \
                        inter_csf * self.probmaps.timings['t2_dash'][2]
            d = inter_gm * self.probmaps.timings['d_map'][0] + \
                    inter_wm * self.probmaps.timings['d_map'][1] + \
                    inter_csf * self.probmaps.timings['d_map'][2]

            mask = pd > 1e-6
            shape = torch.tensor(mask.shape)

            self.optimizers.zero_grad()
            
            preds = []
            targets = []
            sequences = []
            plot_offset_idxs = []

            for sequence in self.sequences:

                sequence.obj_p.PD = pd[mask]
                sequence.obj_p.T1 = t1[mask]
                sequence.obj_p.T2 = t2[mask]
                sequence.obj_p.D = d[mask] * 100000
                sequence.obj_p.T2dash = t2dash[mask]
                sequence.obj_p.B0 = sequence.phantom.B0[mask]
                sequence.obj_p.B1 = sequence.phantom.B1[:, mask]
                sequence.obj_p.coil_sens = sequence.obj_p.original_coil_sens[:, mask]
                sequence.obj_p.nyquist = torch.tensor(
                    shape, device=sequence.obj_p.PD.device
                ) / 2 / sequence.obj_p.size

                if not (sequence.obj_p.PD.shape == sequence.obj_p.T1.shape == \
                        sequence.obj_p.T2.shape == sequence.obj_p.T2dash.shape \
                            == sequence.obj_p.B0.shape):
                    raise Exception("Mismatch of voxel-data shapes")
                if not sequence.obj_p.PD.ndim == 1:
                    raise Exception("Data must be 1D (flattened)")
                if sequence.obj_p.B1.ndim < 2 or sequence.obj_p.B1.shape[1] \
                    != sequence.obj_p.PD.numel():
                    raise Exception("B1 must have shape [coils, voxel_count]")
                if sequence.obj_p.coil_sens.ndim < 2 or \
                    sequence.obj_p.coil_sens.shape[1] != \
                        sequence.obj_p.PD.numel():
                    raise Exception("coil_sens must have shape [coils, voxel_count]")


                sequence.obj_p.avg_B1_trig = sequence.calc_avg_B1_trig(
                    sequence.obj_p.B1, sequence.obj_p.PD
                )

                pos_x, pos_y, pos_z = torch.meshgrid(
                    sequence.obj_p.size[0] * torch.fft.fftshift(
                        torch.fft.fftfreq(int(shape[0]), device=pd.device)
                    ),
                    sequence.obj_p.size[1] * torch.fft.fftshift(
                        torch.fft.fftfreq(int(shape[1]), device=pd.device)
                    ),
                    sequence.obj_p.size[2] * torch.fft.fftshift(
                        torch.fft.fftfreq(int(shape[2]), device=pd.device)
                    ),
                )

                sequence.obj_p.voxel_pos = torch.stack([
                    pos_x[mask].flatten(),
                    pos_y[mask].flatten(),
                    pos_z[mask].flatten()
                ], dim=1)

                predictions = sequence.forward_pass(False)
                if isinstance(predictions, tuple):
                    preds = preds + [pred for pred in predictions]
                    sequences = sequences + ([sequence] * len(predictions))
                    num_plots = sequence.Nc
                    plot_offset_idxs = plot_offset_idxs + \
                            [num_plots * idx for idx in range(len(predictions))]
                else:
                    preds.append(predictions)
                    sequences.append(sequence)
                    plot_offset_idxs.append(0)

                if isinstance(sequence.target, tuple):
                    targets = targets + [tgt for tgt in sequence.target]
                else:
                    targets.append(sequence.target)

            loss = self.loss.calc_loss(targets, preds)
            loss.backward()
            for param in self.probmaps.optimizable_params:
                self.logger.info(
                    f"Max Grad: {param.grad.max()} ; Min grad: {param.grad.min()}"
                )
                param.grad = param.grad + self.sigma * torch.randn_like(param.grad)

            self.optimizers.step(epoch)

            # Save plots and tensors
            if epoch%self.args.save_every==0:
                plots_dir = os.path.join(self.args.plots_dir, f"epoch_{epoch}")
                self.probmaps.save_plots(plots_dir)
                for pred, tgt, sequence, plt_idx in \
                            zip(preds, targets, sequences, plot_offset_idxs):
                    sequence.save_plots(
                        plots_dir, pred, tgt, plt_idx,
                        plot_kspace_difference = \
                            True if self.args.kspace_loss else False
                    )
                self.loss.save_plots(plots_dir)
                if self.args.upsample:
                    self.csf_upsample.save_plots(plots_dir)
                    self.gm_upsample.save_plots(plots_dir)
                    self.wm_upsample.save_plots(plots_dir)

            toc = time.time()
            self.logger.info(f"Time taken to run epoch {epoch}: {toc-tic} sec")
