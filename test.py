from envs.env_dataloader import create_dataloaders
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from envs.new_edit_photo import PhotoEditor
from sac.sac_inference import InferenceAgent
import yaml
from envs.photo_env import PhotoEnhancementEnvTest
import numpy as np

class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def main():
    with open("configs/inference_config.yaml") as f:
        inf_config_dict =yaml.load(f, Loader=yaml.FullLoader)

    inference_config = Config(inf_config_dict)
    photo_editor = PhotoEditor(inference_config.sliders_to_use)

    inference_env = PhotoEnhancementEnvTest(
                        batch_size=inference_config.test_batch_size,
                        imsize=inference_config.imsize,
                        training_mode=False,
                        done_threshold=inference_config.threshold_psnr,
                        pre_encode=False,
                        edit_sliders=inference_config.sliders_to_use,
                        features_size=inference_config.features_size,
                        logger=None
    )

    inf_agent =InferenceAgent(inference_env, inference_config)

    ssim_metric = StructuralSimilarityIndexMeasure()
    test_512 = create_dataloaders(batch_size=1,image_size=64,train=False,pre_encode= False,shuffle=False,resize=False)
    transform = transforms.Compose([
                v2.Resize(size = (64,64), interpolation= transforms.InterpolationMode.BICUBIC),
            ])
    PSNRS = []
    SSIM = []
    for i,t in test_512:
        input = i/255.0
        target = t/255.0 
        parameters = inf_agent.act(transform(input))
        enhanced_image = photo_editor((input.permute(0,2,3,1)).cpu(),parameters[2].cpu())
        psnr = inference_env.compute_rewards(enhanced_image.permute(0,3,1,2),target).item()+50
        ssim = ssim_metric(enhanced_image.permute(0,3,1,2),target).item()
        PSNRS.append(psnr)
        SSIM.append(ssim)
    print(f'Mean PSNR on MIT 5K Dataset {round(np.mean(PSNRS),2)}')
    print(f'Mean SSIM on MIT 5K Dataset {round(np.mean(SSIM),3)}')

if __name__ == "main":
    main()