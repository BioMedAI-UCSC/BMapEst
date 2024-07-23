import numpy as np
import torch
import os

from typing import List, Tuple
from sklearn.decomposition import PCA
from utils.util import generate_prob_map_init_mask, draw, draw_single_image
from utils.metrics import Metrics

class BrainProbabilityMaps:
    def __init__(
        self,
        ground_truth_path: str = None,
        full_data_dir: str = None,
        upsampled_data_dir: str = None,
        metrics_dir: str = None,
        ignore_subjects_list: List = None,
        probmap_shape: Tuple = None,
        init_method: str = "constant",
        init_constant: float = 1e-3,
        interpolated_idx: int = 1,
        init_noise_to_add: str = None,
        init_avg: bool= False,
        optim_csf: bool = True,
        optim_gm: bool = True,
        optim_wm: bool = True,
        pca_coeff_optim: bool = False,
        num_pca_coefficients: int = -1,
        linear_combination_optim: bool = False,
        pixelwise_constant: bool = False,
        coeff_multiplier: float = 1.0,
        logger = None,
        device = "gpu"
    ):
        """
            Parameter
            ---------
                ground_truth_path: str
                    Path of the Brainweb ground truth subject
                full_data_dir: str
                    Path of the Brainweb subjects dataset which has .npz files
                ignore_subjects_list: str
                    Comma separated subject lists to ignore during coefficients
                    optimization
                probmap_shape: List OR Tuple OR torch.shape
                    Shape of the probability maps for optimization
                init_method: str
                    Initialization method
                init_constant: float
                    Float value for 'constant' initialization method
                optim_csf: bool
                    Enables optimization of CSF
                optim_gm: bool
                    Enables optimization of GM
                optim_wm: bool
                    Enables optimization of WM
                linear_combination_optim: bool
                    Enables probability map coefficients optimization
                coeff_multiplier: bool
                    Normalizer that gets multiplied with the coefficients
                logger: Logger
                    Logging object
                device: str
                    Either CPU or GPU
        """
        self.logger = logger
        self.pca_coeff_optim = pca_coeff_optim
        self.num_pca_coefficients = num_pca_coefficients
        self.linear_comb_optim = linear_combination_optim
        self.optimizable_params = []
        self.device = device
        self.optim_csf = optim_csf
        self.optim_gm = optim_gm
        self.optim_wm = optim_wm
        self.pixelwise_constant = pixelwise_constant
        self.init_avg = init_avg
        self.init_noise_to_add = init_noise_to_add
        self.upsampled_data_dir = upsampled_data_dir
        self.metrics_dir = metrics_dir
        self.interpolated_idx = interpolated_idx
        self.save_pca_maps = True
        self.timings = {
            "pd" : [0.8, 0.7, 1.0],
            "t1" : [1.55, 0.83, 4.16],
            "t2" : [0.09, 0.07, 1.65, 0.125],
            "t2_dash" : [0.322, 0.183, 0.0591],
            "d_map" : [0.83, 0.65, 3.19]
        }
        self.subjects = [
                '04', '05', '06', '18', '20',
                '38', '41', '42', '43', '44',
                '45', '46', '47', '48', '49',
                '50', '51', '52', '53', '54'
            ]

        if isinstance(ignore_subjects_list, type(None)):
            self.logger.warning("Default ignoring subject 44")
            ignore_subjects_list = ['44']
        elif isinstance(ignore_subjects_list, str):
            ignore_subjects_list = ignore_subjects_list.split(",")

        self.subjects = list(filter(
            lambda i: i not in ignore_subjects_list, self.subjects
        ))

        if self.linear_comb_optim or self.pca_coeff_optim:
            self.logger.info("Performing linear/PCA coefficients optimization")
            if not os.path.exists(full_data_dir):
                raise NotADirectoryError(full_data_dir)
            self.prepare_params_for_linear_pca_combination_optim(
                ground_truth_path, full_data_dir, coeff_multiplier,
                probmap_shape
            )
        else:
            self.logger.info("Performing probability maps optimization")
            self.prepare_params_for_maps_optim(
                ground_truth_path, probmap_shape, init_method, init_constant,
                full_data_dir
            )
        if not isinstance(upsampled_data_dir, type(None)):
            self.setup_upsampled_gt_maps(ground_truth_path)

        self.metrics_gtdir = os.path.join(self.metrics_dir, 'ground_truths')
        os.makedirs(self.metrics_gtdir, exist_ok=True)
        self.metrics = Metrics(
            self.metrics_dir,
            self.metrics_gtdir,
            ssim=True, psnr=True, l2=False, vgg=False, device="cpu"
        )

        self.save_ground_truth_prob_maps_for_metric_calc()

    def save_ground_truth_prob_maps_for_metric_calc(self):

        if self.optim_csf:
            csf_path = os.path.join(self.metrics_gtdir, "csf.tensor")
            with torch.no_grad():
                torch.save(self.gt_csf, csf_path)
            self.logger.info("Saved ground truth CSF tensor for metrics")

        if self.optim_gm:
            gm_path = os.path.join(self.metrics_gtdir, "gm.tensor")
            with torch.no_grad():
                torch.save(self.gt_gm, gm_path)
            self.logger.info("Saved ground truth GM tensor for metrics")

        if self.optim_wm:
            wm_path = os.path.join(self.metrics_gtdir, "wm.tensor")
            with torch.no_grad():
                torch.save(self.gt_wm, wm_path)
            self.logger.info("Saved ground truth WM tensor for metrics")

    def setup_upsampled_gt_maps(self, ground_truth_path):
        upsampled_gt_map_path = os.path.join(
            self.upsampled_data_dir,
            os.path.split(ground_truth_path)[1]
        )

        data = np.load(upsampled_gt_map_path)
        # CSF
        self.upsampled_tissueCSF = data['tissue_CSF']
        self.upsampled_tissueGM = data['tissue_GM']
        self.upsampled_tissueWM = data['tissue_WM']

    def generate_init_tensor(self, init_method, init_constant, probmap_shape):
        """
            Method to generate a torch tensor based on the initialization
            method. The possible initialization methods could be a 'constant', 
            'uniform', and 'ellipse'.

            Parameter
            ---------
                init_method: str
                    Initialization method
                init_constant: float
                    Float value for 'constant' initialization method
                probmap_shape: Tuple OR List OR torch.shape
                    Shape of the probability maps for optimization
            
            Return
            ------
                torch.Tensor of shape probmap shape
                
        """
        assert len(probmap_shape) in [2, 3], "Shape must have 2 or 3 dims"
        if init_method=="constant":
            self.logger.info("Initializing a constant tensor")
            return torch.ones(probmap_shape, device=self.device) * init_constant
        elif init_method=="uniform":
            self.logger.info("Initializing a tensor from Uniform distribution")
            return torch.rand(probmap_shape, device=self.device)
        elif init_method=="ellipse":
            self.logger.info("Initializing a ellipse shape tensor")
            tensor = generate_prob_map_init_mask(probmap_shape, shape="ellipse")
            return tensor.to(self.device)
        raise NotImplementedError(f"{init_method}")

    def load_map(self, path, shape):
        """
            This method loads the .npz Brainweb file located at path. It also 
            get reshaped to size of shape with the Area as interpolation method.

            Parameter
            ---------
                path: str
                    Location of the Brainweb subject data
                shape: List OR Tuple OR torch.shape
                    Shape for resizing the loaded data
            
            Return
            ------
                torch.Tensor, torch.Tensor, torch.Tensor:
                    CSF, GM, WM
        """
        data = np.load(path)
        # CSF
        tissue_csf  = torch.from_numpy(data['tissue_CSF'])
        tissue_csf = torch.nn.functional.interpolate(
            tissue_csf[None, None, ...],
            size=shape,
            mode="area"
        ).squeeze().unsqueeze(-1)

        # GM
        tissue_gm = torch.from_numpy(data['tissue_GM'])
        tissue_gm = torch.nn.functional.interpolate(
            tissue_gm[None, None, ...],
            size=shape,
            mode="area"
        ).squeeze().unsqueeze(-1)

        # WM
        tissue_wm = torch.from_numpy(data['tissue_WM'])
        tissue_wm = torch.nn.functional.interpolate(
            tissue_wm[None, None, ...],
            size=shape,
            mode="area"
        ).squeeze().unsqueeze(-1)

        return torch.tensor(tissue_csf, device=self.device), \
                torch.tensor(tissue_gm, device=self.device), \
                torch.tensor(tissue_wm, device=self.device)

    def load_interpolated_image(
        self, interpolated_map_path, probmap_shape
    ):
        """
            This method load the interpolated map of the ground truth subject.
            Parameters
            ----------
                data_dir: str
                    Path of the interpolated directory
                probmap_shape: List OR Tuple OR torch.shape
                    Shape for resizing the loaded data
            Return
            ------
                torch.Tensor, torch.Tensor, torch.Tensor
        """
        data = np.load(interpolated_map_path)

        # CSF
        tissue_csf  = torch.from_numpy(data['interploated_csf'])
        tissue_csf = torch.nn.functional.interpolate(
            tissue_csf[None, None, ...],
            size=probmap_shape,
            mode="area"
        ).squeeze().unsqueeze(-1)
        
        # GM
        tissue_gm = torch.from_numpy(data['interploated_gm'])
        tissue_gm = torch.nn.functional.interpolate(
            tissue_gm[None, None, ...],
            size=probmap_shape,
            mode="area"
        ).squeeze().unsqueeze(-1)

        # WM
        tissue_wm = torch.from_numpy(data['interploated_wm'])
        tissue_wm = torch.nn.functional.interpolate(
            tissue_wm[None, None, ...],
            size=probmap_shape,
            mode="area"
        ).squeeze().unsqueeze(-1)

        return torch.tensor(tissue_csf, device=self.device), \
                torch.tensor(tissue_gm, device=self.device), \
                torch.tensor(tissue_wm, device=self.device)

    def load_other_subjects(self, data_dir, probmap_shape):
        """
            This method load the probability maps of all the other subjects.
            Parameters
            ----------
        """
        csfs = [] if self.optim_csf else None
        gms = [] if self.optim_gm else None
        wms = [] if self.optim_wm else None
        for subject in self.subjects:
            csf, gm, wm = self.load_map(
                os.path.join(data_dir, f"subject{subject}.npz"),
                probmap_shape
            )
            if self.optim_csf:
                self.logger.info(f"Loaded CSF for subject {subject}")
                csfs.append(csf)
            if self.optim_gm:
                self.logger.info(f"Loaded GM for subject {subject}")
                gms.append(gm)
            if self.optim_wm:
                self.logger.info(f"Loaded WM for subject {subject}")
                wms.append(wm)
        return csfs, gms, wms

    def prepare_params_for_linear_pca_combination_optim(
        self, ground_truth_path, data_dir, coeff_multiplier, probmap_shape
    ):
        """
            This method builts the trainable coefficients of CSF, GM and WM.
            Parameter
            ---------
                ground_truth_path: str
                    Path of the Brainweb ground truth subject
                data_dir: str
                    Path of the Brainweb subjects dataset which has .npz files
                coeff_multiplier: float
                    Normalizer that gets multiplied with the coefficients
                probmap_shape:
                    Shape of the probability maps for optimization
            Return
            ------
                None
        """
        csfs, gms, wms = self.load_other_subjects(data_dir, probmap_shape)
        gt_csf, gt_gm, gt_wm = self.load_map(ground_truth_path, probmap_shape)

        self.gt_csf = gt_csf
        self.gt_gm = gt_gm
        self.gt_wm = gt_wm

        if self.num_pca_coefficients==-1:
            self.num_pca_coefficients = len(list(self.subjects))
        else:
            self.num_pca_coefficients = min(
                self.num_pca_coefficients, len(list(self.subjects))
            )

        if not self.optim_csf:
            self.logger.info("Initializing CSF from ground truth")
            csfs = gt_csf
        else:
            self.logger.info("Optimizing CSF coefficient tensor")
            if not self.pixelwise_constant:
                self.logger.info("CSF only has a single coefficient optimizer!")
                self.csf_coef = torch.tensor(
                    float(coeff_multiplier),
                    requires_grad=True,
                    device=self.device
                )
            else:
                self.csf_coef = torch.tensor(
                    torch.ones_like(
                        torch.stack(csfs if not self.pca_coeff_optim else
                                        csfs[:self.num_pca_coefficients])
                    ).squeeze() * coeff_multiplier,
                    requires_grad=True,
                    device=self.device
                )
            if self.pca_coeff_optim:
                self.logger.info("PCA CSF component fitting")
                csfs = torch.stack(csfs)
                _, height, width, channels = csfs.shape
                assert channels==1, f"Channels must be of shape 1 but got {channels}"
                csfs = csfs.reshape([csfs.shape[0], height * width])
                csfs_pca_caller = PCA(n_components=self.num_pca_coefficients)
                csfs_pca_caller.fit(csfs)
                csfs = torch.from_numpy(
                    csfs_pca_caller.components_.reshape([
                        self.num_pca_coefficients, height, width
                ])).to(self.device)
            else:
                csfs = torch.stack(csfs).squeeze(-1).to(self.device)
            self.optimizable_params.append(self.csf_coef)
        if not self.optim_gm:
            self.logger.info("Initializing GM from ground truth")
            gms = gt_gm
        else:
            self.logger.info("Optimizing GM coefficient tensor")
            if not self.pixelwise_constant:
                self.logger.info("GM only has a single coefficient optimizer!")
                self.gm_coef = torch.tensor(
                    float(coeff_multiplier),
                    requires_grad=True,
                    device=self.device
                )
            else:
                self.gm_coef = torch.tensor(
                    torch.ones_like(
                        torch.stack(gms if not self.pca_coeff_optim else
                                        gms[:self.num_pca_coefficients])
                    ).squeeze() * coeff_multiplier,
                    requires_grad=True,
                    device=self.device
                )
            if self.pca_coeff_optim:
                self.logger.info("PCA GM component fitting")
                gms = torch.stack(gms)
                _, height, width, channels = gms.shape
                assert channels==1, f"Channels must be of shape 1 but got {channels}"
                gms = gms.reshape([gms.shape[0], height * width])
                gms_pca_caller = PCA(n_components=self.num_pca_coefficients)
                gms_pca_caller.fit(gms)
                gms = torch.from_numpy(
                    gms_pca_caller.components_.reshape([
                        self.num_pca_coefficients, height, width
                ])).to(self.device)
            else:
                gms = torch.stack(gms).squeeze(-1).to(self.device)
            self.optimizable_params.append(self.gm_coef)
        if not self.optim_wm:
            self.logger.info("Initializing WM from ground truth")
            wms = gt_wm
        else:
            self.logger.info("Optimizing WM coefficient tensor")
            if not self.pixelwise_constant:
                self.logger.info("WM only has a single coefficient optimizer!")
                self.wm_coef = torch.tensor(
                    [float(coeff_multiplier)],
                    requires_grad=True,
                    device=self.device
                )
            else:
                self.wm_coef = torch.tensor(
                    torch.ones_like(
                        torch.stack(wms if not self.pca_coeff_optim else
                                        wms[:self.num_pca_coefficients])
                    ).squeeze() * coeff_multiplier,
                    requires_grad=True,
                    device=self.device
                )
            if self.pca_coeff_optim:
                self.logger.info("PCA WM component fitting")
                wms = torch.stack(wms)
                _, height, width, channels = wms.shape
                assert channels==1, f"Channels must be of shape 1 but got {channels}"
                wms = wms.reshape([wms.shape[0], height * width])
                wms_pca_caller = PCA(n_components=self.num_pca_coefficients)
                wms_pca_caller.fit(wms)
                wms = torch.from_numpy(
                    wms_pca_caller.components_.reshape([
                        self.num_pca_coefficients, height, width
                ])).to(self.device)
            else:
                wms = torch.stack(wms).squeeze(-1).to(self.device)
            self.optimizable_params.append(self.wm_coef)

        self.csf = csfs
        self.gm = gms
        self.wm = wms

        csf_shape = self.csf_coef.shape if self.optim_csf else None
        gm_shape = self.gm_coef.shape if self.optim_gm else None
        wm_shape = self.wm_coef.shape if self.optim_wm else None

        self.logger.info(
            f"CSF shape: {self.csf.shape} ; Coef shape: {csf_shape}"
        )
        self.logger.info(f"GM shape: {self.gm.shape} ; Coef shape: {gm_shape}")
        self.logger.info(f"WM shape: {self.wm.shape} ; Coef shape: {wm_shape}")

    def prepare_params_for_maps_optim(
        self, ground_truth_path, probmap_shape, init_method, init_constant,
        data_dir
    ):
        """
            Parameter
            ---------
            ground_truth_path: str
                Path of the Brainweb ground truth subject
            probmap_shape: List OR Tuple OR torch.shape
                Shape of the probability maps for optimization
            init_method: str
                Initialization method
            init_constant: float
                Float value for 'constant' initialization method
            data_dir: str
                    Path of the Brainweb subjects dataset which has .npz files
            Return
            ------
                None
        """
        if not isinstance(ground_truth_path, type(None)) and \
            not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"{ground_truth_path}")

        if isinstance(probmap_shape, type(None)):
            msg = f"Overriding probmap shape from None to {self.csf.shape}"
            self.logger.info(msg)
            probmap_shape = self.csf.shape

        interpolated_file = os.path.join(
            os.path.splitext(ground_truth_path)[0],
            "interpolated_{}.npz".format(self.interpolated_idx)
        )
        csfs, gms, wms = self.load_other_subjects(data_dir, probmap_shape)
        gt_csf, gt_gm, gt_wm = self.load_map(ground_truth_path, probmap_shape)
        if init_method=="interpolated":
            csf_intp, gm_intp, wm_intp = self.load_interpolated_image(
                interpolated_file, probmap_shape=probmap_shape
            )

        self.gt_csf = gt_csf
        self.gt_gm = gt_gm
        self.gt_wm = gt_wm

        # Optimize CSF
        if self.optim_csf:
            self.logger.info("Optimizing CSF map tensor")
            if self.init_avg:
                self.logger.info("Taking average of all CSF maps!")
                self.csf = torch.stack(csfs).squeeze(-1).to(self.device)
                self.csf = torch.mean(self.csf, axis=0).unsqueeze(-1)
            else:
                if init_method=="interpolated":
                    self.csf = csf_intp
                elif init_method=="ground_truth":
                    self.csf = gt_csf.clone()
                else:
                    self.csf = self.generate_init_tensor(
                        init_method, init_constant, probmap_shape
                    )

            self.csf.requires_grad = True
            self.optimizable_params.append(self.csf)
        else:
            self.logger.info("Initializing CSF map from Phantom")
            self.csf = gt_csf

        # Optimize Grey Matter
        if self.optim_gm:
            self.logger.info("Optimizing GM map tensor")
            if self.init_avg:
                self.logger.info("Taking average of all the GM maps!")
                self.gm = torch.stack(gms).squeeze(-1).to(self.device)
                self.gm = torch.mean(self.gm, axis=0).unsqueeze(-1)
            else:
                if init_method=="interpolated":
                    self.gm = gm_intp
                elif init_method=="ground_truth":
                    self.gm = gt_gm.clone()
                else:
                    self.gm = self.generate_init_tensor(
                        init_method, init_constant, probmap_shape
                    )
            self.gm.requires_grad = True
            self.optimizable_params.append(self.gm)
        else:
            self.logger.info("Initializing GM map from Phantom")
            self.gm = gt_gm

        # Optimize White Matter
        if self.optim_wm:
            self.logger.info("Optimizing WM map tensor")
            if self.init_avg:
                self.logger.info("Taking average of all the WM maps!")
                self.wm = torch.stack(wms).squeeze(-1).to(self.device)
                self.wm = torch.mean(self.wm, axis=0).unsqueeze(-1)
            else:
                if init_method=="interpolated":
                    self.wm = wm_intp
                elif init_method=="ground_truth":
                    self.wm = gt_wm.clone().detach()
                else:
                    self.wm = self.generate_init_tensor(
                        init_method, init_constant, probmap_shape
                    )
            self.wm.requires_grad = True
            self.optimizable_params.append(self.wm)
        else:
            self.logger.info("Initializing WM map from Phantom")
            self.wm = gt_wm
    
    def save_plots(self, plots_dir):
        # Metrics directory wrt to an epoch
        metrics_dir = os.path.join(
            self.metrics_dir,
            os.path.split(plots_dir)[1]
        )
        os.makedirs(metrics_dir, exist_ok=True)

        # Plots directory wrt to an epoch
        os.makedirs(plots_dir, exist_ok=True)
        csf_path = os.path.join(plots_dir, "csf.png")
        gm_path = os.path.join(plots_dir, "gm.png")
        wm_path = os.path.join(plots_dir, "wm.png")

        # CSF
        gt_csf = self.gt_csf.detach().cpu().numpy()
        csf = self.csf.detach().cpu()

        # GM
        gm = self.gm.detach().cpu()
        gt_gm = self.gt_gm.detach().cpu().numpy()

        # WM
        wm = self.wm.detach().cpu()
        gt_wm = self.gt_wm.detach().cpu().numpy()

        sigmoid = torch.nn.Sigmoid()

        if self.linear_comb_optim or self.pca_coeff_optim:
            if self.save_pca_maps:
                pca_maps_dir = os.path.join(plots_dir, "pca_maps")
                os.makedirs(pca_maps_dir, exist_ok=True)
            if self.optim_csf:
                coef = self.csf_coef.detach().cpu()
                inter_csf = torch.unsqueeze((csf*coef).sum(axis=0), axis=-1)
                if self.pca_coeff_optim:
                    inter_csf = sigmoid(inter_csf)
                    if self.save_pca_maps:
                        for idx in range(csf.shape[0]):
                            draw_single_image(
                                csf[idx, :, :], image_name=f"csf_{idx}", 
                                clim=None,
                                save_at=os.path.join(
                                                pca_maps_dir, f"csf_{idx}.png"),
                                axis=False
                            )
                else:
                    inter_csf = torch.clamp(inter_csf, min=0.0, max=1.0)
                inter_csf_numpy = inter_csf.numpy()
                draw(gt_csf, inter_csf_numpy, gt_name="CSF_target",
                     pred_name="CSF Predicted", save_at=csf_path)
                csf_metrics = self.metrics.get_metrics(
                    inter_csf, self.gt_csf.detach().cpu(), "CSF"
                )
                self.metrics.update_metrics_list(csf_metrics)
                with torch.no_grad():
                    torch.save(
                        inter_csf,
                        os.path.join(metrics_dir, "csf.tensor")
                    )
            else:
                draw(gt_csf, csf, gt_name="CSF_target", pred_name="FIXED",
                     save_at=csf_path)

            if self.optim_gm:
                coef = self.gm_coef.detach().cpu()
                inter_gm = torch.unsqueeze((gm*coef).sum(axis=0), axis=-1)
                if self.pca_coeff_optim:
                    inter_gm = sigmoid(inter_gm)
                    if self.save_pca_maps:
                        for idx in range(csf.shape[0]):
                            draw_single_image(
                                gm[idx, :, :], image_name=f"gm{idx}",
                                clim=None,
                                save_at=os.path.join(
                                                pca_maps_dir, f"gm{idx}.png"),
                                axis=False
                            )
                else:
                    inter_gm = torch.clamp(inter_gm, min=0.0, max=1.0)
                inter_gm_numpy = inter_gm.numpy()
                draw(gt_gm, inter_gm_numpy, gt_name="GM_target",
                          pred_name="GM Predicted", save_at=gm_path)
                gm_metrics = self.metrics.get_metrics(
                    inter_gm, self.gt_gm.detach().cpu(), "GM"
                )
                self.metrics.update_metrics_list(gm_metrics)
                with torch.no_grad():
                    torch.save(inter_gm, os.path.join(metrics_dir, "gm.tensor"))
            else:
                draw(gt_gm, gm, gt_name="GM_target", pred_name="FIXED",
                          save_at=gm_path)

            if self.optim_wm:
                coef = self.wm_coef.detach().cpu()
                inter_wm = torch.unsqueeze((wm*coef).sum(axis=0), axis=-1)
                if self.pca_coeff_optim:
                    inter_wm = sigmoid(inter_wm)
                    if self.save_pca_maps:
                        for idx in range(csf.shape[0]):
                            draw_single_image(
                                wm[idx, :, :], image_name=f"wm{idx}",
                                clim=None,
                                save_at=os.path.join(
                                                pca_maps_dir, f"wm{idx}.png"),
                                axis=False
                            )
                else:
                    inter_wm = torch.clamp(inter_wm, min=0.0, max=1.0)
                inter_wm_numpy = inter_wm.numpy()
                draw(gt_wm, inter_wm_numpy, gt_name="WM_target",
                     pred_name="WM Prediction", save_at=wm_path)
                wm_metrics = self.metrics.get_metrics(
                    inter_wm, self.gt_wm.detach().cpu(), "WM"
                )
                self.metrics.update_metrics_list(wm_metrics)
                with torch.no_grad():
                    torch.save(inter_wm, os.path.join(metrics_dir, "wm.tensor"))
            else:
                draw(gt_wm, wm, gt_name="WM_target", pred_name="FIXED",
                     save_at=wm_path)
            self.save_pca_maps = False

        else:
            csf[torch.isnan(csf)] = 0
            csf = torch.clamp(csf, min=0, max=1)
            draw(
                gt_csf, csf, gt_name="CSF_target",  save_at=csf_path,
                pred_name="CSF_Predicted" if self.optim_csf else "FIXED"
            )
            if self.optim_csf:
                csf_metrics = self.metrics.get_metrics(
                    csf, self.gt_csf.detach().cpu(), "CSF"
                )
                self.metrics.update_metrics_list(csf_metrics)
                with torch.no_grad():
                    torch.save(csf, os.path.join(metrics_dir, "csf.tensor"))
            gm[torch.isnan(gm)] = 0
            gm = torch.clamp(gm, min=0.0, max=1.0)
            draw(
                gt_gm, gm, gt_name="GM_target", save_at=gm_path,
                pred_name="GM_Predicted" if self.optim_gm else "FIXED"
            )
            if self.optim_gm:
                gm_metrics = self.metrics.get_metrics(
                    gm, self.gt_gm.detach().cpu(), "GM"
                )
                self.metrics.update_metrics_list(gm_metrics)
                with torch.no_grad():
                    torch.save(gm, os.path.join(metrics_dir, "gm.tensor"))
            wm[torch.isnan(wm)] = 0
            wm = torch.clamp(wm, min=0.0, max=1.0)
            draw(
                gt_wm, wm, gt_name="WM_target", save_at=wm_path,
                pred_name="WM_Predicted" if self.optim_wm else "FIXED"
            )
            if self.optim_wm:
                wm_metrics = self.metrics.get_metrics(
                    wm, self.gt_wm.detach().cpu(), "WM"
                )
                self.metrics.update_metrics_list(wm_metrics)
                with torch.no_grad():
                    torch.save(wm, os.path.join(metrics_dir, "wm.tensor"))
        self.metrics.flush_metrics_list(
            os.path.join(metrics_dir, "runtime_metrics.csv")
        )