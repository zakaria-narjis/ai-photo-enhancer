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
from tqdm import tqdm
class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path', help='YAML config file')
    parser.add_argument('--logger_level', type=int, default=logging.INFO)
    args = parser.parse_args()
    logger = logging.getLogger("test")
    
    # Configure logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logger_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(args.logger_level)

    with open("configs/inference_config.yaml") as f:
        inf_config_dict =yaml.load(f, Loader=yaml.FullLoader)

    inference_config = Config(inf_config_dict)
    photo_editor = PhotoEditor(inference_config.sliders_to_use)

    inference_env = PhotoEnhancementEnvTest(
                        batch_size=inference_config.batch_size,
                        imsize=inference_config.imsize,
                        training_mode=False,
                        done_threshold=inference_config.threshold_psnr,
                        pre_encode=False,
                        edit_sliders=inference_config.sliders_to_use,
                        features_size=inference_config.features_size,
                        logger=None
    )

    inf_agent =InferenceAgent(inference_env, inference_config)

    inf_agent.load_backbone(args.models_path+'backbone.pth')
    inf_agent.load_actor_weights(args.models_path+'actor_head.pth')
    inf_agent.load_critics_weights(args.models_path+'qf1_head.pth',args.models_path+'qf2_head.pth')

    ssim_metric = StructuralSimilarityIndexMeasure()
    test_512 = create_dataloaders(batch_size=1,image_size=64,train=False,pre_encode= False,shuffle=False,resize=False)
    transform = transforms.Compose([
                v2.Resize(size = (64,64), interpolation= transforms.InterpolationMode.BICUBIC),
            ])
    PSNRS = []
    SSIM = []
    logger.info(f'Testing ...')
    for i,t in tqdm(test_512):
        input = i/255.0
        target = t/255.0 
        parameters = inf_agent.act(transform(input))
        enhanced_image = photo_editor((input.permute(0,2,3,1)).cpu(),parameters[2].cpu())
        psnr = inference_env.compute_rewards(enhanced_image.permute(0,3,1,2),target).item()+50
        ssim = ssim_metric(enhanced_image.permute(0,3,1,2),target).item()
        PSNRS.append(psnr)
        SSIM.append(ssim)
    logger.info(f'Mean PSNR on MIT 5K Dataset {round(np.mean(PSNRS),2)}')
    logger.info(f'Mean SSIM on MIT 5K Dataset {round(np.mean(SSIM),3)}')

if __name__ == "__main__":
    main()