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
from streamlit import cache_resource
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# Set page config to wide mode
st.set_page_config(layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
MODEL_PATH = (
    "experiments/runs/ResNet_10_sliders__224_128_aug__2024-07-23_21-23-35"
)
SLIDERS = [
    "temp",
    "tint",
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "vibrance",
    "saturation",
]
SLIDERS_ORD = [
    "contrast",
    "exposure",
    "temp",
    "tint",
    "whites",
    "blacks",
    "highlights",
    "shadows",
    "vibrance",
    "saturation",
]


class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


@cache_resource
def load_preprocessor_agent(preprocessor_agent_path, device):
    # with open(
    #     os.path.join(preprocessor_agent_path, "configs/sac_config.yaml")
    # ) as f:
    #     sac_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(
        os.path.join(preprocessor_agent_path, "configs/env_config.yaml")
    ) as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join("src/configs/inference_config.yaml")) as f:
        inf_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    inference_config = Config(inf_config_dict)
    # sac_config = Config(sac_config_dict)
    env_config = Config(env_config_dict)

    inference_env = PhotoEnhancementEnvTest(
        batch_size=env_config.train_batch_size,
        imsize=env_config.imsize,
        training_mode=None,
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
    )

    inference_config.device = device
    preprocessor_agent = InferenceAgent(inference_env, inference_config)
    preprocessor_agent.device = device
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
    return preprocessor_agent


enhancer_agent = load_preprocessor_agent(MODEL_PATH, DEVICE)
photo_editor = PhotoEditor(SLIDERS)


def enhance_image(image: np.array, params: dict):
    input_image = image.unsqueeze(0).to(DEVICE)
    parameters = [params[param_name] / 100.0 for param_name in SLIDERS_ORD]
    parameters = torch.tensor(parameters).unsqueeze(0).to(DEVICE)
    enhanced_image = photo_editor(input_image, parameters)
    enhanced_image = enhanced_image.squeeze(0).cpu().detach().numpy()
    enhanced_image = np.clip(enhanced_image, 0, 1)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return enhanced_image


def auto_enhance(image, deterministic=True):
    input_image = image.unsqueeze(0).to(DEVICE)
    input_image = input_image.permute(0, 3, 1, 2)
    IMAGE_SIZE = enhancer_agent.env.imsize
    input_image = F.resize(
        input_image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=F.InterpolationMode.BICUBIC,
    )
    batch_observation = TensorDict(
        {
            "batch_images": input_image,
        },
        batch_size=[input_image.shape[0]],
    )
    parameters = enhancer_agent.act(
        batch_observation, deterministic=deterministic, n_samples=0
    )
    parameters = parameters.squeeze(0) * 100.0
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


def slider_callback():
    for name in SLIDERS:
        st.session_state.params[name] = st.session_state[f"slider_{name}"]
    image_tensor = (
        torch.from_numpy(st.session_state.original_image).float() / 255.0
    )
    st.session_state.enhanced_image = enhance_image(
        image_tensor, st.session_state.params
    )


def auto_random_enhance_callback():
    image_tensor = (
        torch.from_numpy(st.session_state.original_image).float() / 255.0
    )
    auto_params = auto_enhance(image_tensor, deterministic=False)
    for i, name in enumerate(SLIDERS_ORD):
        st.session_state[f"slider_{name}"] = int(auto_params[i])
        st.session_state.params[name] = int(auto_params[i])
    st.session_state.enhanced_image = enhance_image(
        image_tensor, st.session_state.params
    )


def auto_enhance_callback():
    image_tensor = (
        torch.from_numpy(st.session_state.original_image).float() / 255.0
    )
    auto_params = auto_enhance(image_tensor)
    for i, name in enumerate(SLIDERS_ORD):
        st.session_state[f"slider_{name}"] = int(auto_params[i])
        st.session_state.params[name] = int(auto_params[i])
    st.session_state.enhanced_image = enhance_image(
        image_tensor, st.session_state.params
    )


def reset_sliders():
    for name in SLIDERS:
        st.session_state[f"slider_{name}"] = 0
        st.session_state.params[name] = 0
    # st.session_state.enhanced_image = enhance_image(image_tensor, st.session_state.params) # noqa: E501
    st.session_state.enhanced_image = st.session_state.original_image


def reset_on_upload():
    st.session_state.original_image = None
    reset_sliders()


def create_smooth_histogram(image):
    # Compute histograms for each channel # noqa: E501
    bins = np.linspace(0, 255, 256)
    hist_r, _ = np.histogram(image[..., 0], bins=bins)
    hist_g, _ = np.histogram(image[..., 1], bins=bins)
    hist_b, _ = np.histogram(image[..., 2], bins=bins)

    # Normalize the histograms
    def normalize_histogram(hist):
        hist_central = hist[1:-1]
        hist_max = np.max(hist_central)
        hist_min = np.min(hist_central)

        hist_normalized = (hist_central - hist_min) / (hist_max - hist_min)

        hist[0] = min(hist[0] / hist_max, 1)
        hist[-1] = min(hist[-1] / hist_max, 1)

        return np.concatenate(([hist[0]], hist_normalized, [hist[-1]]))

    hist_r_norm = normalize_histogram(hist_r)
    hist_g_norm = normalize_histogram(hist_g)
    hist_b_norm = normalize_histogram(hist_b)

    # Create Bokeh figure with transparent background
    p = figure(
        width=300,
        height=150,
        toolbar_location=None,
        x_range=(0, 255),
        y_range=(0, 1.1),
        background_fill_color=None,
        border_fill_color=None,
        outline_line_color=None,
    )

    # Remove all axes, labels, and grids
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Create ColumnDataSource for each channel
    source_r = ColumnDataSource(
        data=dict(left=bins[:-1], right=bins[1:], top=hist_r_norm)
    )
    source_g = ColumnDataSource(
        data=dict(left=bins[:-1], right=bins[1:], top=hist_g_norm)
    )
    source_b = ColumnDataSource(
        data=dict(left=bins[:-1], right=bins[1:], top=hist_b_norm)
    )

    # Plot the histograms
    p.quad(
        bottom=0,
        top="top",
        left="left",
        right="right",
        source=source_r,
        fill_color="red",
        fill_alpha=0.9,
        line_color=None,
    )
    p.quad(
        bottom=0,
        top="top",
        left="left",
        right="right",
        source=source_g,
        fill_color="green",
        fill_alpha=0.9,
        line_color=None,
    )
    p.quad(
        bottom=0,
        top="top",
        left="left",
        right="right",
        source=source_b,
        fill_color="blue",
        fill_alpha=0.9,
        line_color=None,
    )

    # Remove padding
    p.min_border_left = 0
    p.min_border_right = 0
    p.min_border_top = 0
    p.min_border_bottom = 0

    return p


# In your Streamlit app
def plot_histogram_streamlit(image):
    histogram = create_smooth_histogram(image)
    st.sidebar.bokeh_chart(histogram, use_container_width=True)


# Initialize session state
if "enhanced_image" not in st.session_state:
    st.session_state.enhanced_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "params" not in st.session_state:
    st.session_state.params = {name: 0 for name in SLIDERS}
for name in SLIDERS:
    if f"slider_{name}" not in st.session_state:
        st.session_state[f"slider_{name}"] = 0

# Set up the Streamlit app
st.title("Photo Enhancement App")

# File uploader in the main area
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", ".tif"],
    on_change=reset_on_upload,
)

if uploaded_file is not None:
    # Load the original image
    st.session_state.original_image = np.array(
        Image.open(uploaded_file).convert("RGB"), dtype=np.uint16
    )

    # Enhance the image initially
    if st.session_state.enhanced_image is None:
        st.session_state.enhanced_image = st.session_state.original_image

    # Sidebar for controls
    st.sidebar.title("Controls")

    # Display histogram
    st.sidebar.subheader("Colors Histogram")
    plot_histogram_streamlit(st.session_state.enhanced_image)

    # Select box to choose which image to display
    display_option = st.sidebar.selectbox(
        "Select view mode", ("Comparison", "Enhanced")
    )

    # Create two columns for the buttons
    col1, col2, col3 = st.sidebar.columns(3)

    # Button for auto-enhancement
    with col1:
        st.button(
            "Auto Enhance",
            on_click=auto_enhance_callback,
            key="auto_enhance_button",
            use_container_width=True,
        )

    with col2:
        st.button(
            "Auto Random Enhance",
            on_click=auto_random_enhance_callback,
            key="auto_random_enhance_button",
            use_container_width=True,
        )
    # Button for resetting sliders
    with col3:
        st.button(
            "Reset",
            on_click=reset_sliders,
            key="reset_button",
            use_container_width=True,
        )

    st.sidebar.subheader("Adjustments")
    slider_names = SLIDERS

    for name in slider_names:
        if f"slider_{name}" not in st.session_state:
            st.session_state[f"slider_{name}"] = 0

        st.sidebar.slider(
            name.capitalize(),
            min_value=-100,
            max_value=100,
            value=st.session_state[f"slider_{name}"],
            key=f"slider_{name}",
            on_change=slider_callback,
        )

    # Create a single column to maximize width
    left_spacer, content_column, right_spacer = st.columns([1, 3, 1])
    with content_column:
        if display_option == "Enhanced":
            if st.session_state.enhanced_image is not None:
                st.image(
                    st.session_state.enhanced_image.astype(np.uint8),
                    caption="Enhanced Image",
                    use_column_width=True,
                )
            else:
                st.warning(
                    "Enhanced image is not available. Try adjusting the sliders or clicking 'Auto Enhance'."  # noqa: E501
                )
        else:  # Comparison view
            if st.session_state.enhanced_image is not None:
                image_comparison(
                    img1=Image.fromarray(
                        st.session_state.original_image.astype(np.uint8)
                    ),
                    img2=Image.fromarray(
                        st.session_state.enhanced_image.astype(np.uint8)
                    ),
                    label1="Original",
                    label2="Enhanced",
                    width=850,  # You might want to adjust this value
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                )
            else:
                st.warning(
                    "Enhanced image is not available for comparison. Try adjusting the sliders or clicking 'Auto Enhance'."  # noqa: E501
                )

    # Add custom CSS to make the image comparison component responsive
    st.markdown(
        """
    <style>
    .stImageComparison {
        width: 100% !important;
    }
    .stImageComparison > figure > div {
        width: 100% !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
