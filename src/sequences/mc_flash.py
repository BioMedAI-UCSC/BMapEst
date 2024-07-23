import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
import os

import matplotlib.pyplot as plt
import torch.nn as nn

from src.sequences import Sequences
from utils.util import sigmoid, gaussian_kernel

class GaussianLayer(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, kernel_size=5):
        super(GaussianLayer, self).__init__()
        self.layer = nn.Conv2d(
            in_chan, out_chan, kernel_size, stride=1, 
            padding=0, bias=None, groups=out_chan
        )
        self.layer.weight.requires_grad = False
        self.layer.weight.data = gaussian_kernel(
            l=kernel_size, out_chan=out_chan
        )
    def forward(self, x):
        return self.layer(x)

class McFlash(Sequences):
    seq_types = [
        "T1inv", "T2prep", "T2star", "DIRprep", "FLAIRprep", "diff_prep"
    ]
    def __init__(self, args, logger, **kwargs):
        super().__init__(args, logger)
        logger.info("Using McFlash sequences!")
        self.experiment_id = 'McFlash_' + kwargs['sequence_type']
        self.sequence_type = kwargs['sequence_type']
        self.Nc = kwargs['num_contrasts']
        self.gaussian_smoothing = GaussianLayer(
            in_chan=self.Nc, out_chan=self.Nc, kernel_size=5
        )
        self.sequence_index = f"{self.seq_types.index(self.sequence_type)}"
        self.build()

    def build(self):
        self.system = pp.Opts(
            max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
            rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
            grad_raster_time=1e-5)
        self.fov = 200e-3
        self.slice_thickness = 8e-3
        self.build_sequence()

        self.seq = self.build_sequence()
        self.check_timings()

        # TODO: Check this part
        self.seq.set_definition(
            'FOV', [self.fov, self.fov, self.slice_thickness]
        )
        self.seq.set_definition('Name', 'FLASH')
        seq_filepath = os.path.join(
            self.args.output, self.experiment_id + '.seq'
        )
        self.seq.write(seq_filepath)
        self.seq = mr0.Sequence.import_file(seq_filepath)
        if self.args.device=="cuda":
            self.logger.info("Moved the Sequence into GPU!")
            self.seq = self.seq.cuda()
    
    def save_gt_plots(self, plots_dir):
        gt_dir = os.path.join(plots_dir, "ground_truths")
        os.makedirs(gt_dir, exist_ok=True)
        output =  "{}_plot.png"
        for ii in range(0, self.Nc):
            if self.args.kspace_smoothing:
                plt.subplot(1, 4, ii+1); plt.title(self.sequence_type)
                plt.imshow(np.rot90(
                    np.abs(self.target[0][ii, :, :])
                ), cmap="gray")
                plt.subplot(2, 4, ii+1); plt.title(self.sequence_type)
                plt.imshow(np.rot90(
                    np.abs(self.target[1][ii, :, :])
                ), cmap="gray")
            else:
                plt.subplot(1, 4, ii+1); plt.title(self.sequence_type)
                plt.imshow(np.rot90(np.abs(self.target[ii, :, :])), cmap="gray")
        plt.show()
        plt.savefig(os.path.join(
            gt_dir, output.format(self.sequence_type)
        ))

    def build_sequence(self):
        """
            This method is specific to the FLASH sequence.
        """
        seq = pp.Sequence()
        FA_flash=5

        # Define rf events
        rf1, _, _ = pp.make_sinc_pulse(
            flip_angle= FA_flash * np.pi / 180, duration=1e-3,
            slice_thickness=self.slice_thickness, apodization=0.5, 
            time_bw_product=4, system=self.system, return_gz=True
        )

        rf90 = pp.make_block_pulse(
           flip_angle=90 * np.pi / 180, duration=1e-3, system=self.system
        )

        rf180 = pp.make_block_pulse(
           flip_angle=180 * np.pi / 180, phase_offset=90 * np.pi/180,
           duration=1e-3, system=self.system
        )

        rf90_ = pp.make_block_pulse(
           flip_angle=90 * np.pi / 180, phase_offset=180 * np.pi/180,
           duration=1e-3, system=self.system
        )

        dwell= 3e-5
        # Define other gradients and ADC events
        gx = pp.make_trapezoid(
           channel='x', flat_area=self.Nread/self.fov,
           flat_time=self.Nread*dwell, system=self.system
        )
        adc = pp.make_adc(
           num_samples=self.Nread, duration=self.Nread*dwell,
           phase_offset=0 * np.pi/180, delay=gx.rise_time, system=self.system
        )
        gx_pre = pp.make_trapezoid(
           channel='x', area=-gx.area/2, duration=1e-3, system=self.system
        )
        gx_spoil = pp.make_trapezoid(
           channel='x', area=1.5*gx.area, duration=1.5e-3, system=self.system
        )

        gz_spoil = pp.make_trapezoid(
           channel='z', area=1.5*gx.area, duration=1.5e-3, system=self.system
        )

        #McFLash contrast arrays
        TI      = [0.1, 0.5, 1,  2.7]
        TI2     = [0.1, 0.5, 1,  3]
        TEprep  = [0.01, 0.05, 0.1,  0.3]
        TEd     = [0.005, 0.01, 0.03,  0.06]
        DG      = [10, 50, 100, 300]*1000000
        self.logger.info("Sequence type is: " + self.sequence_type)

        for ii in range(0, self.Nc):
            # ======
            # CONSTRUCT PREPARATION
            # ======
            match self.sequence_type:
                case 'none': # no prep
                    seq.add_block(pp.make_delay(0.001))
                    break
                case 'T1inv': # inv recovery
                    seq.add_block(rf180)
                    seq.add_block(pp.make_delay(TI[ii]))
                case 'T2prep':
                    seq.add_block(rf90)
                    seq.add_block(pp.make_delay(round(TEprep[ii]/2*1e6)/1e6))
                    seq.add_block(rf180)
                    seq.add_block(pp.make_delay(round(TEprep[ii]/2*1e6)/1e6))
                    seq.add_block(rf90_)
                case 'FLAIRprep':  # FLAIR is fluid suppressed T2-weigted
                    seq.add_block(rf180)
                    seq.add_block(pp.make_delay(2.7))
                    seq.add_block(rf90)
                    seq.add_block(pp.make_delay(round(TEprep[ii]/2*1e6)/1e6))
                    seq.add_block(rf180)
                    seq.add_block(pp.make_delay(round(TEprep[ii]/2*1e6)/1e6))
                    seq.add_block(rf90_)
                case 'DIRprep':  # Double inversion recovery T2
                    seq.add_block(rf180)
                    seq.add_block(pp.make_delay(TI[ii]))
                    seq.add_block(rf180)
                    seq.add_block(pp.make_delay(0.5))
                case 'diff_prep':
                    # This is a DWI gradient
                    gx_diff = pp.make_trapezoid(
                        channel='x', area=DG[ii], duration=10e-3,
                        system=self.system
                    )
                    seq.add_block(rf90)
                    seq.add_block(gx_diff)
                    seq.add_block(rf180)
                    seq.add_block(gx_diff)
                    seq.add_block(rf90_)
                case 'T2star':
                    self.logger.info('FLASH TE altered')
                case _:
                    self.logger.info("Unknown McFlash: self.sequence_type")
            seq.add_block(gz_spoil)

            # ======
            # CONSTRUCT READOUT SEQUENCE  rf and gradient spoiled centric flash
            # ======
            rf_phase = 0
            rf_inc   = 0
            rf_spoiling_inc = 50

            ##linear reordering
            phenc = np.arange(-self.Nphase // 2, self.Nphase // 2, 1)
            ## centric reordering
            self.permvec = sorted(
                np.arange(len(phenc)), key=lambda x: abs(len(phenc) // 2 - x)
            )
            phenc_centr = phenc[self.permvec]

            for jj in range(0, self.Nphase):  # e.g. -64:63
                # set current rf phase
                rf1.phase_offset = rf_phase / 180 * np.pi

                # follow with ADC
                adc.phase_offset = rf_phase / 180 * np.pi
                seq.add_block(rf1)

                # increase increment
                rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]

                # increment additional pahse
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
                if self.sequence_type=='T2star':
                    seq.add_block(pp.make_delay(TEd[ii]))

                gp = pp.make_trapezoid(
                    channel='y', area=phenc_centr[jj]/self.fov,
                    duration=1e-3, system=self.system
                )
                seq.add_block(gx_pre, gp)
                seq.add_block(adc, gx)
                gp = pp.make_trapezoid(
                    channel='y', area=-phenc_centr[jj]/self.fov,
                    duration=1e-3, system=self.system
                )
                seq.add_block(gx_spoil, gp)
            seq.add_block(pp.make_delay(10))
        return seq

    def forward_pass(self, build_gt=False):
        """
            Method to perform the MRI simulation using the MRTwin simulator.
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
        signal = self.run_mrzero_pipeline()

        if self.args.signal_loss and build_gt:
            self.logger.info("Setting up signal as target variable!")
            self.target = torch.abs(signal).clone()
            return None
        elif self.args.signal_loss:
            return signal

        kspace = torch.reshape(
            (signal), (self.Nc, self.Nphase, self.Nread)
        ).clone().permute(0, 2, 1)

        ipermvec = np.argsort(self.permvec)

        kspace=kspace[:,:,ipermvec]

        if self.args.kspace_loss and not self.args.signal_loss and build_gt:
            self.logger.info("Setting up kspace as target variable!")
            kspace_abs = torch.abs(kspace).clone()
            if self.args.kspace_smoothing:
                self.target = (kspace_abs, self.gaussian_smoothing(kspace_abs))
            else:
                self.target = kspace_abs
            return None
        elif self.args.kspace_loss:
            result = torch.abs(kspace)
            if self.args.kspace_smoothing:
                return (result, self.gaussian_smoothing(result))
            else:
                return result
        
        x = torch.fft.ifftshift(kspace, dim=(1, 2))
        x = torch.fft.fft2(x, dim=(1, 2))
        img = torch.fft.fftshift(x,dim=(1, 2))

        if not self.args.kspace_loss and not self.args.signal_loss and build_gt:
            self.logger.info("Setting up absolute space as target variable!")
            self.target = torch.abs(img).clone()
            if not isinstance(self.filepath, type(None)):
                subject_id = os.path.split(self.filepath)[1].replace(".npz", "")
                out_path = os.path.join(
                    self.args.save_gen_data,
                    f"seq_{self.sequence_index}_{subject_id}.pt"
                )
                with torch.no_grad():
                    torch.save(self.target, out_path)
            return None

        return torch.abs(img)

    def setup_ground_truth(self, filepath=None):
        self.logger.info("Setting up the target variable!")
        self.filepath = filepath
        phantom = mr0.VoxelGridPhantom.brainweb(
            self.args.ground_truth if isinstance(filepath, type(None)) else
                                                                        filepath
        )
        phantom = phantom.interpolate(self.Nread, self.Nphase, 1)

        # TODO: Add the following dephasing func in other sequences
        phantom.dephasing_func = sigmoid
        
        # Manipulate loaded data
        phantom.D *= 100000
        
        # Convert Phantom into simulation data
        self.obj_p = phantom.build()
        
        # TODO: Add the following coil_sensitivity addition in other sequences
        self.obj_p.original_coil_sens = phantom.coil_sens.clone()
        if self.args.device=="cuda":
            self.logger.info("Moved the probmaps into GPU!")
            self.obj_p = self.obj_p.cuda()

        self.phantom = phantom
        self.logger.info("Build the object phantom successfully!")
        self.forward_pass(build_gt=True)

        if isinstance(self.target, type(None)):
            raise RuntimeError("Target variable can't be None!")
        # else:
        #     self.save_gt_plots(self.args.plots_dir)

    def generate_training_data(self):

        all_files = [filename for filename in os.listdir(self.args.data)
                     if os.path.splitext(filename)[1]==".npz"]
        
        for idx, file in enumerate(all_files):
            self.logger.info(
                f"{self.sequence_type}:{idx+1}/{len(all_files)} " + \
                    "files generated!"
            )
            self.logger.info(f"Filename: {file}")
            self.setup_ground_truth(os.path.join(self.args.data, file))
