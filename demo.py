import streamlit as st
import torch
from PIL import Image
import numpy as np
from streamlit_image_comparison import image_comparison
from src.envs.new_edit_photo import PhotoEditor
from src.sac.sac_inference import InferenceAgent
import yaml
import os
from src.envs.photo_env import PhotoEnhancementEnvTest
from tensordict import TensorDict
import torchvision.transforms.v2.functional as F

# Set page config to wide mode
st.set_page_config(layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "experiments/runs/ResNet_6s__ResNet_224_aug__2024-07-22_14-52-08"
SLIDERS = ['temp','tint','exposure', 'contrast','highlights','shadows', 'whites', 'blacks','vibrance','saturation']
SLIDERS_ORD = ['contrast','exposure','temp','tint','whites','blacks','highlights','shadows','vibrance','saturation']

class Config(object):
    def __init__(self,dictionary):
        self.__dict__.update(dictionary)

def load_preprocessor_agent(preprocessor_agent_path,device):
    with open(os.path.join(preprocessor_agent_path,"configs/sac_config.yaml")) as f:
        sac_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(preprocessor_agent_path,"configs/env_config.yaml")) as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join("src/configs/inference_config.yaml")) as f:
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
        use_txt_features=env_config.use_txt_features if hasattr(env_config,'use_txt_features') else False,
        augment_data=False,
        pre_encoding_device=device,
        pre_load_images=False,
        logger=None
    )
    
    inference_config.device = device
    preprocessor_agent = InferenceAgent(inference_env, inference_config)
    preprocessor_agent.device = device
    preprocessor_agent.load_backbone(os.path.join(preprocessor_agent_path,'models','backbone.pth'))
    preprocessor_agent.load_actor_weights(os.path.join(preprocessor_agent_path,'models','actor_head.pth'))
    preprocessor_agent.load_critics_weights(os.path.join(preprocessor_agent_path,'models','qf1_head.pth'),
                                            os.path.join(preprocessor_agent_path,'models','qf2_head.pth'))
    return preprocessor_agent

enhancer_agent = load_preprocessor_agent(MODEL_PATH,DEVICE)
photo_editor = PhotoEditor(SLIDERS)

def enhance_image(image:np.array, params:dict):
    input_image = image.unsqueeze(0).to(DEVICE)
    parameters = [params[param_name]/100.0 for param_name in SLIDERS_ORD]
    parameters = torch.tensor(parameters).unsqueeze(0).to(DEVICE)
    enhanced_image = photo_editor(input_image,parameters)
    enhanced_image = enhanced_image.squeeze(0).cpu().detach().numpy()
    enhanced_image = np.clip(enhanced_image, 0, 1)
    enhanced_image = (enhanced_image*255).astype(np.uint8)
    return enhanced_image

def auto_enhance(image):
    input_image = image.unsqueeze(0).to(DEVICE)
    input_image = input_image.permute(0,3,1,2)
    IMAGE_SIZE = enhancer_agent.env.imsize
    input_image = F.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=F.InterpolationMode.BICUBIC)
    batch_observation = TensorDict(
        {
            "batch_images":input_image,
        },
        batch_size = [input_image.shape[0]],
    )
    parameters = enhancer_agent.act(batch_observation,deterministic=True)
    parameters = parameters.squeeze(0)*100.0
    parameters = torch.round(parameters)
    output_parameters = []
    index = 0
    for slider in SLIDERS_ORD:
        if slider in enhancer_agent.env.edit_sliders:
            output_parameters.append(parameters[index].item())
            index += 1
        else:
            output_parameters.append(0)
    return output_parameters

# Initialize session state
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'params' not in st.session_state:
    st.session_state.params = {name: 0 for name in SLIDERS}

# Set up the Streamlit app
st.title("Photo Enhancement App")

# File uploader in the main area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the original image
    original_image = np.array(Image.open(uploaded_file))
    image_tensor = torch.from_numpy(original_image).float() / 255.0

    # Sidebar for controls
    st.sidebar.title("Controls")

    # Select box to choose which image to display
    display_option = st.sidebar.selectbox(
        "Select view mode",
        ("Original", "Enhanced", "Comparison")
    )

    st.sidebar.subheader("Adjustments")
    slider_names = SLIDERS
    
    for name in slider_names:
        st.session_state.params[name] = st.sidebar.slider(name.capitalize(), min_value=-100, max_value=100, value=st.session_state.params[name])

    # Button for auto-enhancement
    if st.sidebar.button("Auto Enhance"):
        auto_params = auto_enhance(image_tensor)
        for i, name in enumerate(SLIDERS_ORD):
            st.session_state.params[name] = int(auto_params[i])
            

    # Enhance the image based on current parameters
    st.session_state.enhanced_image = enhance_image(image_tensor, st.session_state.params)

    # Create a single column to maximize width
    col1, = st.columns(1)

    # Display the selected image or comparison in the main area
    with col1:
        if display_option == "Original":
            st.image(original_image, caption="Original Image", use_column_width=True)
        elif display_option == "Enhanced":
            st.image(st.session_state.enhanced_image, caption="Enhanced Image", use_column_width=True)
        else:  # Comparison view
            image_comparison(
                img1=Image.fromarray(original_image),
                img2=Image.fromarray(st.session_state.enhanced_image),
                label1="Original",
                label2="Enhanced",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
            )

    # Add custom CSS to make the image comparison component responsive
    st.markdown("""
    <style>
    .stImageComparison {
        width: 100% !important;
    }
    .stImageComparison > figure > div {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
