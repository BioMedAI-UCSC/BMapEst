import argparse
import os

def parse_args():
    """
        Method to parse the command line parameters.
        Parameter
        ---------
        Return
        ------
            args: argparse.ArgumentParser
                Parsed argument object
    """
    parser = argparse.ArgumentParser(
        description="Script to run Probability Map Optimization"
    )
    parser.add_argument("--exp", "-e", type=str, required=True,
                        help="Experiment ID")
    parser.add_argument("--method", "-m", type=str, default="linear",
                        choices=['linear', 'normal'],
                        help="Defines the optimization method")
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="Path to store the results")
    parser.add_argument("--data", "-d", type=str, default="data",
                        help="Path to the data directory")
    parser.add_argument("--upsampled_data", type=str, default=None,
                        help="Path to the upsampled data directory")
    parser.add_argument("--ground_truth", type=str, default="subject44.npz",
                        help="Brainweb filename of ground truth")
    parser.add_argument("--sequences", "-s", type=str, nargs='+',
                        help="Sequence name")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Places simulation either on CPU or GPU")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save results after a certain number of epochs")
    parser.add_argument("--sigma", type=float, default=2,
                        help="Grad noise sigma")

    # Optimization flags
    parser.add_argument("--epochs", type=int, default=400,
                        help="Number of epochs")
    parser.add_argument("--nread", type=int, default=64,
                        help="Number of frequencies")
    parser.add_argument("--nphase", type=int, default=64,
                        help="Number of phases")
    parser.add_argument("--alternate_optim", action="store_true",
                        help="Performs alternate optimization of maps")
    parser.add_argument("--switch_every", type=int, default=40,
                        help="Number of epochs/map in alternate optimization")
    parser.add_argument("--optimize_csf", action="store_true",
                        help="Flag to optimize CSF")
    parser.add_argument("--optimize_wm", action="store_true",
                        help="Flag to optimize White Matter")
    parser.add_argument("--optimize_gm", action="store_true",
                        help="Flag to optimize GM")
    parser.add_argument("--interpolated_idx", type=int, default=None,
                        help="Interpolation index for the subject")
    parser.add_argument("--init_method", type=str, default="constant",
                        choices=["constant", "uniform", "ellipse",
                                 "ground_truth", "interpolated"],
                        help="Map initialization technique")
    parser.add_argument("--init_noise", type=str, default=None,
                        choices=[None, "gaussian", "s&p", "poisson", "speckle"],
                        help="Type of noise to add to the init tensor")
    parser.add_argument("--init_constant", type=float, default=1e-3,
                        help="Initilization constant")
    parser.add_argument("--init_lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--init_avg", action="store_true",
                        help="Initializes probability maps with the avg")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=['adam', 'adamw', 'sgd'],
                        help="Supported optimizers")

    # Loss parameters
    parser.add_argument("--losses", type=str, default="mse", nargs='+',
                        help="Loss types")
    parser.add_argument("--signal_loss", action="store_true",
                        help="Takes loss directly from the signal")
    parser.add_argument("--kspace_loss", action="store_true",
                        help="Takes loss by comapring the k-space")
    parser.add_argument("--kspace_smoothing", action="store_true",
                        help="Augments kspace loss by k-space smoothing loss")
    parser.add_argument("--space_abs_loss", action="store_true",
                        help="Takes loss from the absolution space values")

    # PCA coefficient optimization
    parser.add_argument("--pca_coeff_optim", action="store_true",
                        help="Performs PCA coefficients optimization")
    parser.add_argument("--num_pca_coeff", type=int, default=-1,
                        help="Number of coefficients in PCA optimization")

    # Linear combination optimization
    parser.add_argument("--linear_comb_optim", action="store_true",
                        help="Performs linear combination optimization")
    parser.add_argument("--pixelwise", action="store_true",
                        help="Initialize constant for every pixel")
    parser.add_argument("--ignore_sub", type=str, default="44",
                        help="Comma separated Brainweb subjects id list")
    parser.add_argument("--multiplier", type=float, default=1/19,
                        help="Multipler for coefficient maps")

    # Upsampling approach
    parser.add_argument("--upsample", action="store_true",
                        help="Upsamples the probmaps to 2x!")
    
    # Multi contrast flash sequence specific parameter
    parser.add_argument("--flash_seq_num", "-f", type=int, default=-1,
                        help="Sequence number of Mc FLASH")

    # Parameters to generate the training data
    parser.add_argument("--generate_training_data", action="store_true",
                        help="Generates training data for sequences")

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.data, args.ground_truth)) and \
        not args.generate_training_data:
        raise FileNotFoundError(f"Ground truth {args.ground_truth} is missing")

    if args.signal_loss==args.kspace_loss==args.space_abs_loss==False:
        args.space_abs_loss = True

    if args.optimize_csf==args.optimize_gm==args.optimize_wm==False:
        args.optimize_csf = args.optimize_gm = args.optimize_wm = True
    
    if args.upsample and isinstance(args.upsampled_data, type(None)):
        raise TypeError("Upsampled data directory can't be None")

    args.exp = f"exp_{args.exp}"
    args.ground_truth = os.path.join(args.data, args.ground_truth)
    args.output = os.path.join(args.output, args.exp)
    args.video_out = os.path.join(args.output, "videos")
    args.plots_dir = os.path.join(args.output, "plots")
    args.metrics = os.path.join(args.output, "metrics")
    args.save_gen_data = os.path.join(args.output, "generated_data")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.video_out, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.metrics, exist_ok=True)
    os.makedirs(args.save_gen_data, exist_ok=True)
    return args
