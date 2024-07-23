import torch

class Optimizers():
    def __init__(self, args, logger, probmaps):
        """Class to setup PyTorch Optimizers"""
        self.args = args
        self.logger = logger
        self.probmaps = probmaps
        self.optimizer = None
        if self.args.alternate_optim:
            self.setup_alternate_optimizers()
        else:
            params = [{
                'params' : self.probmaps.optimizable_params,
                'lr' : self.args.init_lr
            }]
            if self.args.optimizer=="adam":
                self.optimizer = torch.optim.Adam(params)
            elif self.args.optimizer=="sgd":
                self.optimizer = torch.optim.SGD(params)

    
    def setup_alternate_optimizers(self):
        """
            This method sets the optimizers of CSF, GM, and WM individually.
            Parameter
            ---------
            Return
            ------
                None
        """
        # CSF
        self.optimize_idx = 0
        self.param_str_list = []
        self.optimizer_csf_switch = self.args.optimize_csf
        if self.args.optimize_csf:
            self.param_str_list.append("optimizer_csf_switch")
            param = self.probmaps.csf_coef if self.args.linear_comb_optim \
                                                        else self.probmaps.csf
            params = [{'params' : param, 'lr' : self.args.init_lr}]
            if self.args.optimizer=="adam":
                self.optimizer_csf = torch.optim.Adam(params)
            elif self.args.optimizer=="sgd":
                self.optimizer_csf = torch.optim.SGD(params)

        # GM
        self.optimizer_gm_switch = self.args.optimize_gm
        if self.args.optimize_gm:
            self.param_str_list.append("optimizer_gm_switch")
            param = self.probmaps.gm_coef if self.args.linear_comb_optim \
                                                        else self.probmaps.gm
            params = [{'params' : param, 'lr' : self.args.init_lr}]
            if self.args.optimizer=="adam":
                self.optimizer_gm = torch.optim.Adam(params)
            elif self.args.optimizer=="sgd":
                self.optimizer_gm = torch.optim.SGD(params)

        # WM
        self.optimizer_wm_switch = self.args.optimize_wm
        if self.args.optimize_wm:
            self.param_str_list.append("optimizer_wm_switch")
            param = self.probmaps.wm_coef if self.args.linear_comb_optim \
                                                        else self.probmaps.wm
            params = [{'params' : param, 'lr' : self.args.init_lr}]
            if self.args.optimizer=="adam":
                self.optimizer_wm = torch.optim.Adam(params)
            elif self.args.optimizer=="sgd":
                self.optimizer_wm = torch.optim.SGD(params)
    
    def zero_grad(self):
        """Zero-out gradients based on the optimization technique"""
        if self.args.alternate_optim:
            if self.optimizer_csf_switch:
                self.optimizer_csf.zero_grad()
            if self.optimizer_gm_switch:
                self.optimizer_gm.zero_grad()
            if self.optimizer_wm_switch:
                self.optimizer_wm.zero_grad()
        else:
            self.optimizer.zero_grad()

    def step(self, epoch):
        """
            Implements the step process of the optimizer based on the 
            optimization technique.
        """
        if self.args.alternate_optim:
            if epoch%self.args.switch_every==0:
                self.optimize_idx = (self.optimize_idx+1)%sum([
                    self.args.optimize_csf, self.args.optimize_gm, 
                    self.args.optimize_wm
                ])
                for param_idx, param_str in enumerate(self.param_str_list):
                    if param_idx==self.optimize_idx:
                        setattr(self, param_str, True)
                    else:
                        setattr(self, param_str, False)
                msg = f"Epoch Optimization Switch: {epoch} -> "
                msg += f"Opt_CSF: {self.optimizer_csf_switch} ; "
                msg += f"Opt_GM: {self.optimizer_gm_switch} ; "
                msg += f"Opt_WM: {self.optimizer_wm_switch}"
                self.logger.info(msg)

            if self.optimizer_csf_switch:
                self.optimizer_csf.step()
            if self.optimizer_gm_switch:
                self.optimizer_gm.step()
            if self.optimizer_wm_switch:
                self.optimizer_wm.step()
        else:
            self.optimizer.step()