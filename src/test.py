from envs.env_dataloader import create_dataloaders
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from envs.new_edit_photo import PhotoEditor
from sac.sac_inference import InferenceAgent
import yaml
from envs.photo_env import PhotoEnhancementEnvTest
import numpy as np
import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch


class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def load_preprocessor_agent(preprocessor_agent_path, device):
    current_dir = Path(__file__).parent.absolute()
    with open(
        os.path.join(preprocessor_agent_path, "configs/sac_config.yaml")
    ) as f:
        sac_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(
        os.path.join(preprocessor_agent_path, "configs/env_config.yaml")
    ) as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(
        os.path.join(current_dir, "../configs/inference_config.yaml")
    ) as f:
        inf_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    inference_config = Config(inf_config_dict)
    sac_config = Config(sac_config_dict)
    env_config = Config(env_config_dict)
    inference_env = PhotoEnhancementEnvTest(
        batch_size=env_config.train_batch_size,
        imsize=env_config.imsize,
        training_mode=False,
        done_threshold=env_config.threshold_psnr,
        edit_sliders=env_config.sliders_to_use,
        features_size=env_config.features_size,
        discretize=env_config.discretize,
        discretize_step=env_config.discretize_step,
        use_txt_features=(
            env_config.use_txt_features
            if hasattr(env_config, "use_txt_features")
            else False
        ),
        augment_data=False,
        pre_encoding_device=device,
        pre_load_images=False,
        logger=None,
    )  # useless just to get the action space size for the Networks and whether to use txt features or not
    preprocessor_photo_editor = PhotoEditor(env_config.sliders_to_use)
    inference_config.device = device
    preprocessor_agent = InferenceAgent(inference_env, inference_config)
    preprocessor_agent.device = device
    os.path.join(preprocessor_agent_path, "models", "backbone.pth")
    preprocessor_agent.load_backbone(
        os.path.join(preprocessor_agent_path, "models", "backbone.pth")
    )
    preprocessor_agent.load_actor_weights(
        os.path.join(preprocessor_agent_path, "models", "actor_head.pth")
    )
    preprocessor_agent.load_critics_weights(
        os.path.join(preprocessor_agent_path, "models", "qf1_head.pth"),
        os.path.join(preprocessor_agent_path, "models", "qf2_head.pth"),
    )
    return preprocessor_agent, preprocessor_photo_editor


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    current_dir = Path(__file__).parent.absolute()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_path", help="folder containing the experiment models"
    )
    parser.add_argument(
        "--deterministic", type=str2bool, nargs="?", const=True, default=False
    )
    # parser.add_argument('--pre_load_images', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--logger_level", type=int, default=logging.INFO)
    parser.add_argument("--device", nargs="?", type=str, default="cuda:0")
    parser.add_argument("--plt_samples", nargs="?", type=int, default=3)
    args = parser.parse_args()
    logger = logging.getLogger("test")
    args.device = (
        torch.device(args.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # Configure logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logger_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(args.logger_level)

    with open(os.path.join(current_dir, "configs/inference_config.yaml")) as f:
        inf_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(
        os.path.join(args.experiment_path, "configs/sac_config.yaml")
    ) as f:
        sac_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(
        os.path.join(args.experiment_path, "configs/env_config.yaml")
    ) as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    inference_config = Config(inf_config_dict)
    sac_config = Config(sac_config_dict)
    env_config = Config(env_config_dict)
    if hasattr(env_config, "preprocessor_agent_path") == False:
        env_config.preprocessor_agent_path = None
    SEED = sac_config.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = sac_config.torch_deterministic
    torch.autograd.set_detect_anomaly(True)

    inference_config.device = args.device
    photo_editor = PhotoEditor(env_config.sliders_to_use)

    inference_env = PhotoEnhancementEnvTest(
        batch_size=env_config.train_batch_size,
        imsize=env_config.imsize,
        training_mode=False,
        done_threshold=env_config.threshold_psnr,
        edit_sliders=env_config.sliders_to_use,
        features_size=env_config.features_size,
        discretize=env_config.discretize,
        discretize_step=env_config.discretize_step,
        use_txt_features=(
            env_config.use_txt_features
            if hasattr(env_config, "use_txt_features")
            else False
        ),
        augment_data=(
            env_config.augment_data
            if hasattr(env_config, "augment_data")
            else False
        ),
        pre_encoding_device=args.device,
        pre_load_images=False,
        preprocessor_agent_path=None,
        logger=None,
    )  # useless just to get the action space size for the Networks and whether to use txt features or not
    inf_agent = InferenceAgent(inference_env, inference_config)
    os.path.join(args.experiment_path, "models", "backbone.pth")
    inf_agent.load_backbone(
        os.path.join(args.experiment_path, "models", "backbone.pth")
    )
    inf_agent.load_actor_weights(
        os.path.join(args.experiment_path, "models", "actor_head.pth")
    )
    inf_agent.load_critics_weights(
        os.path.join(args.experiment_path, "models", "qf1_head.pth"),
        os.path.join(args.experiment_path, "models", "qf2_head.pth"),
    )

    if env_config.preprocessor_agent_path is not None:
        preprocessor_agent, preprocessor_photo_editor = (
            load_preprocessor_agent(
                env_config.preprocessor_agent_path, args.device
            )
        )
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)

    test_512 = create_dataloaders(
        batch_size=1,
        image_size=env_config.imsize,
        use_txt_features=False,
        train=False,
        augment_data=False,
        shuffle=False,
        resize=False,
        pre_encoding_device=args.device,
        pre_load_images=False,
    )
    test_resized = create_dataloaders(
        batch_size=500,
        image_size=env_config.imsize,
        use_txt_features=(
            env_config.use_txt_features
            if hasattr(env_config, "use_txt_features")
            else False
        ),
        train=False,
        augment_data=False,
        shuffle=False,
        resize=True,
        pre_encoding_device=args.device,
        pre_load_images=True,
    )

    PSNRS = []
    SSIM = []

    logger.info(f"Testing ...")
    logger.info(f"Using device {args.device}")
    # batch_64_images = next(iter(test_64))[0]/255.0
    inference_env.dataloader = test_resized
    inference_env.iter_dataloader = iter(test_resized)
    inference_env.batch_size = 500
    batch_images = inference_env.reset()
    logger.info(
        f"Computing optimal enhancement sliders values, DETERMINISTIC:{args.deterministic}"
    )
    if env_config.preprocessor_agent_path is not None:
        pre_parameters = preprocessor_agent.act(
            batch_images, deterministic=args.deterministic
        )
        preprocessed_images = preprocessor_photo_editor(
            batch_images.permute(0, 2, 3, 1), pre_parameters
        )
        preprocessed_images = preprocessed_images.permute(0, 3, 1, 2)
    else:
        preprocessed_images = batch_images
    parameters = inf_agent.act(
        preprocessed_images, deterministic=args.deterministic
    )

    logger.info(f"Done")
    parameter_counter = 0
    logger.info(f"Enhancing images and computing metrics")

    plot_data = []
    random_indices = random.sample(range(len(test_512)), args.plt_samples)
    for i, t in tqdm(test_512, position=0, leave=True):
        source = i / 255.0
        target = t / 255.0
        if env_config.preprocessor_agent_path is not None:
            enhanced_image = source.permute(0, 2, 3, 1)
            enhanced_image = preprocessor_photo_editor(
                enhanced_image.to(args.device),
                pre_parameters[parameter_counter].unsqueeze(0).to(args.device),
            )
        else:
            enhanced_image = source.permute(0, 2, 3, 1)
        enhanced_image = photo_editor(
            enhanced_image.to(args.device),
            parameters[parameter_counter].unsqueeze(0).to(args.device),
        )
        enhanced_image = enhanced_image.permute(0, 3, 1, 2)  # B,C,H,W
        psnr = (
            inference_env.compute_rewards(
                enhanced_image.to(args.device), target.to(args.device)
            ).item()
            + 50
        )
        ssim = ssim_metric(
            enhanced_image.to(args.device), target.to(args.device)
        ).item()
        PSNRS.append(psnr)
        SSIM.append(ssim)
        if parameter_counter in random_indices:
            enhanced_image = enhanced_image.permute(0, 2, 3, 1)  # B,H,W,C
            plot_data.append(
                (source.cpu(), enhanced_image.cpu(), target.cpu(), psnr, ssim)
            )
        parameter_counter += 1

    mean_PSNRS = round(np.mean(PSNRS), 2)
    mean_SSIM = round(np.mean(SSIM), 3)
    logger.info(f"Mean PSNR on MIT 5K Dataset {mean_PSNRS}")
    logger.info(f"Mean SSIM on MIT 5K Dataset {mean_SSIM}")

    # Plotting

    fig, axes = plt.subplots(
        3, args.plt_samples, figsize=(15, args.plt_samples * 5)
    )
    # plt.subplots_adjust(hspace=0.5)
    logger.info(f"Plotting samples")
    for index in range(args.plt_samples):
        plot_data[index][0]
        axes[0][index].imshow(plot_data[index][0][0].permute(1, 2, 0))
        # axes[0][0].set_title(('source_img'))
        axes[0][index].axis("off")
        axes[1][index].imshow(plot_data[index][1][0])
        # axes[1][index].set_title('Ours')
        axes[1][index].axis("off")
        axes[1][index].text(
            0.5,
            -0.04,
            f"PSNR:{round(plot_data[index][3],2)}, SSIM:{round(plot_data[index][4],2)}",
            size=10,
            ha="center",
            transform=axes[1][index].transAxes,
        )
        axes[2][index].imshow(plot_data[index][2][0].permute(1, 2, 0))
        axes[2][index].axis("off")
    plt.tight_layout()
    logger.info(
        f'Saving plot in {os.path.join(args.experiment_path,"samples_plot.svg")}'
    )
    fig.savefig(
        os.path.join(args.experiment_path, "samples_plot.svg"), format="svg"
    )


if __name__ == "__main__":
    main()
