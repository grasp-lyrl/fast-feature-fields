import os
import yaml
import wandb
import logging

import torch
import torch._dynamo

try:
    import torchsparse # type: ignore
except ImportError:
    torchsparse = None
    print("torchsparse not installed. Sparse operations will not work.")

from accelerate import Accelerator

from .utils_gen import generate_alphanumeric


def setup_torch(cudnn_benchmark: bool=False):
    torch.manual_seed(403)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = cudnn_benchmark # turn on for faster training if we are using the fixed event mode
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.compiled_autograd = True
    if torchsparse is not None:
        torchsparse.backends.benchmark = True


def setup_experiment(args, base_path: str, models_path: str):
    """
        Takes in the arguments and sets up the required files, folders and logging for the experiment.
        Also sets up wandb if desired.
    """
    assert args.train["batch"] % args.train["mini_batch"] == 0, "train_batch should be divisible by mini_batch"

    resume = os.path.exists(f"{models_path}/last.pth")
    if not resume:
        os.makedirs(base_path, exist_ok=True)
        for dir in ["predictions", "models", "training_events"]:
            os.makedirs(f"{base_path}/{dir}", exist_ok=True)

    logging.basicConfig(filename=f"{base_path}/exp.log", filemode="a",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {args.name}, with SLURM jobid: {os.environ.get('SLURM_JOB_ID', None)}")

    if args.wandb:
        if resume:
            wandb_run_id = yaml.safe_load(open(f"{base_path}/args.yml", "r"))["wandb_run_id"]
            wandb.init(project="f3", config=args, name=args.name, id=wandb_run_id, resume="must")
        else:
            wandb.init(project="f3", config=args, name=args.name)
            args.wandb_run_id = wandb.run.id
        logger.info(f"Logging to wandb with name: {args.name} and run_id: {wandb.run.id}")

    if not resume:
        with open(f"{base_path}/args.yml", "w") as f:
            yaml.dump(vars(args), f, default_flow_style=None)
    
    return logger, resume


def setup_accelerate_experiment(args, base_path: str, models_path: str):
    """
        Takes in the arguments and sets up the required files, folders and logging for the experiment.
        Also sets up wandb if desired. Accelerate version.
    """
    ##################### Accelerate Setup #####################
    gradient_accumulation_steps = args.train["batch"] // args.train["mini_batch"]
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                              log_with="wandb" if args.wandb else None)
    device = accelerator.device
    
    assert args.train["batch"] % args.train["mini_batch"] == 0, "train_batch should be divisible by mini_batch"

    resume = os.path.exists(f"{models_path}/last.pth")
    if not resume:
        os.makedirs(base_path, exist_ok=True)
        for dir in ["predictions", "models", "training_events"]:
            os.makedirs(f"{base_path}/{dir}", exist_ok=True)

    logging.basicConfig(filename=f"{base_path}/exp.log", filemode="a",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {args.name}, with SLURM jobid: {os.environ.get('SLURM_JOB_ID', None)}")

    if args.wandb:
        if resume:
            args.wandb_run_id = yaml.safe_load(open(f"{base_path}/args.yml", "r"))["wandb_run_id"]
            accelerator.init_trackers(project_name="f3", config=args,
                                      init_kwargs={"wandb": {"name": args.name, "id": args.wandb_run_id, "resume": "must"}})
        else:
            args.wandb_run_id = generate_alphanumeric(7)
            accelerator.init_trackers(project_name="f3", config=args,
                                      init_kwargs={"wandb": {"name": args.name, "id": args.wandb_run_id}})
        logger.info(f"Logging to wandb with name: {args.name} and run_id: {args.wandb_run_id}")

    if not resume:
        with open(f"{base_path}/args.yml", "w") as f:
            yaml.dump(vars(args), f, default_flow_style=None)

    return logger, resume, accelerator, device, gradient_accumulation_steps
