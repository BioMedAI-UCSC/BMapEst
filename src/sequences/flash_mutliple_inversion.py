from src.sequences import Sequences
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
import os

class FlashMultipleInversion(Sequences):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        logger.info("Using Flash Sequence with multiple inversion recovery!")
        self.experiment_id = 'FLASH_2D_FIT'
        # self.ti = [0.1, 0.5, 1.0, 5.0]
        self.ti = [0.1, 0.2, 0.3, 0.4, 0.5] # Best
        logger.info(f"Flash TI values: {','.join(list(map(str, self.ti)))}")
        self.build()

    def build(self):
        self.system = pp.Opts(
            max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
            rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
            grad_raster_time=50*10e-6
        )
        fov = 1000e-3
        self.slice_thickness = 8e-3
        self.phenc = np.arange(-self.Nphase // 2, self.Nphase // 2, 1)
        ## centric reordering
        self.permvec = sorted(
            np.arange(len(self.phenc)),
            key=lambda x: abs(len(self.phenc) // 2 - x)
        )
        self.seq = self.create_inv_rec()
        self.check_timings()
        self.seq.set_definition(
            'FOV', [fov, fov, self.slice_thickness]
        )
        self.seq.set_definition('Name', 'FLASH')
        seq_filepath = os.path.join(
            self.args.output, self.experiment_id + '.seq'
        )
        self.seq.write(seq_filepath)
        self.seq = mr0.Sequence.from_seq_file(seq_filepath)
        if self.args.device=="cuda":
            self.logger.info("Moved the Sequence into GPU!")
            self.seq = self.seq.cuda()

    def create_inv_rec(self):
        """
            This method is specific to the FLASH sequence.
        """
        seq = pp.Sequence()

        # Define rf events
        rf1, _, _ = pp.make_sinc_pulse(
            flip_angle=10 * np.pi / 180, duration=1e-3,
            slice_thickness=self.slice_thickness, apodization=0.5, 
            time_bw_product=4, system=self.system, return_gz=True
        )

        rf_inv = pp.make_block_pulse(
            flip_angle=180 * np.pi / 180, duration=1e-3, system=self.system
        )

        # Define other gradients and ADC events
        gx = pp.make_trapezoid(
            channel='x', flat_area=self.Nread, 
            flat_time=4e-3, system=self.system
        )
        adc = pp.make_adc(
            num_samples=self.Nread, duration=4e-3, phase_offset=0 * np.pi/180, 
            delay=gx.rise_time, system=self.system
        )
        gx_pre = pp.make_trapezoid(
            channel='x', area=-gx.area / 2, duration=1e-3, system=self.system
        )
        gx_spoil = pp.make_trapezoid(
            channel='x', area=1.5 * gx.area, duration=2e-3, system=self.system
        )
        gy_spoil = pp.make_trapezoid(
            channel='y', area=1.5 * gx.area, duration=2e-3, system=self.system
        )

        rf_phase = 0
        rf_inc = 0
        rf_spoiling_inc = 117

        # ======
        # CONSTRUCT SEQUENCE
        # ======

        phenc_centr = self.phenc[self.permvec]
        for t in self.ti:
            seq.add_block(rf_inv)
            seq.add_block(pp.make_delay(t))
            seq.add_block(gx_spoil, gy_spoil)
            for ii in range(0, self.Nphase):  # e.g. -64:63
                # set current rf phase
                rf1.phase_offset = rf_phase / 180 * np.pi

                # follow with ADC
                adc.phase_offset = rf_phase / 180 * np.pi

                # increase increment
                rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]

                # increment additional pahse
                rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

                seq.add_block(rf1)
                seq.add_block(pp.make_delay(0.005))

                gp = pp.make_trapezoid(
                    channel='y', area=phenc_centr[ii],
                    duration=1e-3, system=self.system
                )
                seq.add_block(gx_pre, gp)
                seq.add_block(adc, gx)

                gp = pp.make_trapezoid(
                    channel='y', area=-phenc_centr[ii],
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

        ipermvec = np.argsort(self.permvec)
        kspace = torch.reshape(
            (signal), (len(self.ti), self.Nphase, self.Nread)
        ).clone().permute(0, 2, 1)
        kspace=kspace[:,:,ipermvec]

        if self.args.kspace_loss and not self.args.signal_loss and build_gt:
            self.logger.info("Setting up kspace as target variable!")
            self.target = torch.abs(kspace).clone()
            return None
        elif self.args.kspace_loss:
            return torch.abs(kspace)

        # fftshift
        spectrum = torch.fft.fftshift(kspace, dim=(1, 2))
        # FFT
        space = torch.fft.ifft2(spectrum)
        # fftshift
        space = torch.fft.ifftshift(space, dim=(1, 2))

        if not self.args.kspace_loss and not self.args.signal_loss and build_gt:
            self.logger.info("Setting up absolute space as target variable!")
            self.target = torch.abs(space).clone()
            return None

        return torch.abs(space)

    def setup_ground_truth(self):
        self.logger.info("Setting up the target variable!")
        phantom = mr0.VoxelGridPhantom.brainweb(self.args.ground_truth)
        phantom = phantom.interpolate(self.Nread, self.Nread, 1)
        # Manipulate loaded data
        phantom.T2dash[:] = 30e-3
        phantom.D *= 0
        phantom.B0 *= 1    # alter the B0 inhomogeneity
        # Convert Phantom into simulation data
        self.obj_p = phantom.build()
        if self.args.device=="cuda":
            self.logger.info("Moved the probmaps into GPU!")
            self.obj_p = self.obj_p.cuda()

        self.phantom = phantom
        self.logger.info("Build the object phantom successfully!")
        self.forward_pass(build_gt=True)

        if isinstance(self.target, type(None)):
            raise RuntimeError("Target variable can't be None!")
