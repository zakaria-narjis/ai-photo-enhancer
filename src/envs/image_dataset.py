import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2.functional as F
import os
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
from tqdm import tqdm
import kornia.color as K


class FiveKDataset(Dataset):
    def __init__(
        self,
        image_size,
        mode="train",
        resize=True,
        augment_data=False,
        use_txt_features=False,
        device="cuda",
        pre_load_images=True,
    ):
        current_dir = os.getcwd()
        dataset_dir = os.path.join(current_dir, "dataset")
        self.IMGS_PATH = os.path.join(dataset_dir, f"FiveK/{mode}")
        self.FEATURES_PATH = os.path.join(
            dataset_dir, "processed_categories_2.txt"
        )
        self.resize = resize
        self.image_size = image_size
        self.augment_data = augment_data
        self.use_txt_features = use_txt_features
        self.device = device
        self.pre_load_images = pre_load_images
        self.feature_categories = ["Location", "Time", "Light", "Subject"]

        # Load semantic features from processed_categories.txt
        self.features = {}
        with open(self.FEATURES_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                img_name = parts[0]
                img_features = parts[1:]
                self.features[img_name] = img_features

        # Load image files
        self.img_files = [
            f for f in os.listdir(os.path.join(self.IMGS_PATH, "input"))
        ]

        # Prepare MultiLabelBinarizer
        all_features = []
        for img in self.img_files:
            original_img = img.replace("_duplicated", "")
            all_features.append(self.features[original_img])

        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(all_features)

        # Create encoding dictionaries for categorical approach
        unique_features = {
            cat: set(feat[i] for feat in all_features)
            for i, cat in enumerate(self.feature_categories)
        }
        self.feature_to_idx = {
            cat: {feat: idx for idx, feat in enumerate(sorted(features))}
            for cat, features in unique_features.items()
        }
        if self.use_txt_features == "histogram":
            self.histogram_bins = 64
        # Preload all images and features if pre_load_images is True
        if self.pre_load_images:
            self.preload_data()

        if self.use_txt_features == "embedded":
            self.precompute_features()

    def compute_histogram(self, image):
        # Convert RGB to CIE Lab
        lab_image = K.rgb_to_lab(image.unsqueeze(0) / 255.0).squeeze(0)

        # Compute histogram for each channel
        histograms = []
        for channel in range(3):  # L, a, b channels
            if channel == 0:  # L channel
                hist = torch.histc(
                    lab_image[channel],
                    bins=self.histogram_bins,
                    min=0,
                    max=100,
                )
            else:  # a and b channels
                hist = torch.histc(
                    lab_image[channel],
                    bins=self.histogram_bins,
                    min=-128,
                    max=127,
                )
            # Normalize the histogram
            hist = hist / hist.sum()
            histograms.append(hist)

        # Concatenate histograms
        return torch.cat(histograms)

    def preload_data(self):
        print("Preloading images and features...")
        self.source_images = []
        self.target_images = []
        self.one_hot_features = []
        self.cat_features = []
        self.histograms = []

        for img_name in tqdm(self.img_files):
            # Load and preprocess images
            source = read_image(
                os.path.join(self.IMGS_PATH, "input", img_name)
            )
            target = read_image(
                os.path.join(self.IMGS_PATH, "target", img_name)
            )

            if self.resize:
                source = F.resize(
                    source,
                    (self.image_size, self.image_size),
                    interpolation=F.InterpolationMode.BICUBIC,
                )
                target = F.resize(
                    target,
                    (self.image_size, self.image_size),
                    interpolation=F.InterpolationMode.BICUBIC,
                )

            self.source_images.append(source.to(self.device))
            self.target_images.append(target.to(self.device))

            # Precompute features
            if self.use_txt_features == "one_hot":
                one_hot = self.mlb.transform([self.features[img_name]])[0]
                self.one_hot_features.append(
                    torch.tensor(
                        one_hot, dtype=torch.float32, device=self.device
                    )
                )
            elif self.use_txt_features == "categorical":
                cat = [
                    self.feature_to_idx[cat][feat]
                    for cat, feat in zip(
                        self.feature_categories, self.features[img_name]
                    )
                ]
                self.cat_features.append(
                    torch.tensor(cat, dtype=torch.long, device=self.device)
                )
            elif self.use_txt_features == "histogram":
                source_hist = self.compute_histogram(source).to(self.device)
                target_hist = self.compute_histogram(target).to(self.device)
                self.histograms.append((source_hist, target_hist))

        self.source_images = torch.stack(self.source_images)
        self.target_images = torch.stack(self.target_images)

        if self.use_txt_features == "one_hot":
            self.one_hot_features = torch.stack(self.one_hot_features)
        elif self.use_txt_features == "categorical":
            self.cat_features = torch.stack(self.cat_features)
        elif self.use_txt_features == "histogram":
            self.source_histograms = torch.stack(
                [h[0] for h in self.histograms]
            )
            self.target_histograms = torch.stack(
                [h[1] for h in self.histograms]
            )

        print("Images and features preloaded and stored in GPU memory.")

    def precompute_features(self):
        print("Precomputing BERT and CLIP features...")
        self.bert_features = []
        self.clip_features = []

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = (
            BertModel.from_pretrained("bert-base-uncased")
            .to(self.device)
            .eval()
        )
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        clip_model = (
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            .to(self.device)
            .eval()
        )

        for img_name in tqdm(self.img_files):
            # Precompute BERT features
            feature_text = " ".join(self.features[img_name])
            inputs = tokenizer(
                feature_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            bert_features = outputs.last_hidden_state[:, 0, :].squeeze(
                0
            )  # Shape: (768,)
            self.bert_features.append(bert_features)

            # Precompute CLIP features
            image = self.source_images[
                len(self.bert_features) - 1
            ].cpu()  # Get the corresponding preloaded image
            clip_inputs = clip_processor(images=image, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                clip_features = clip_model.get_image_features(**clip_inputs)
            self.clip_features.append(
                clip_features.squeeze(0)
            )  # Shape: (512,)

        self.bert_features = torch.stack(self.bert_features).to(self.device)
        self.clip_features = torch.stack(self.clip_features).to(self.device)

        del bert_model, tokenizer, clip_model, clip_processor
        print("BERT and CLIP features precomputed and stored in GPU memory.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if self.pre_load_images:
            source = self.source_images[idx]
            target = self.target_images[idx]
        else:
            img_name = self.img_files[idx]
            source = read_image(
                os.path.join(self.IMGS_PATH, "input", img_name)
            )
            target = read_image(
                os.path.join(self.IMGS_PATH, "target", img_name)
            )

            if self.resize:
                source = F.resize(
                    source,
                    (self.image_size, self.image_size),
                    interpolation=F.InterpolationMode.BICUBIC,
                )
                target = F.resize(
                    target,
                    (self.image_size, self.image_size),
                    interpolation=F.InterpolationMode.BICUBIC,
                )

            source = source.to(self.device)
            target = target.to(self.device)

        if self.augment_data:
            if torch.rand(1).item() > 0.5:
                source = F.hflip(source)
                target = F.hflip(target)
            if torch.rand(1).item() > 0.5:
                source = F.vflip(source)
                target = F.vflip(target)

        if not self.use_txt_features:
            return source, target
        elif self.use_txt_features == "one_hot":
            if self.pre_load_images:
                return source, self.one_hot_features[idx], target
            else:
                one_hot = self.mlb.transform(
                    [self.features[self.img_files[idx]]]
                )[0]
                return (
                    source,
                    torch.tensor(
                        one_hot, dtype=torch.float32, device=self.device
                    ),
                    target,
                )
        elif self.use_txt_features == "categorical":
            if self.pre_load_images:
                return source, self.cat_features[idx], target
            else:
                cat = [
                    self.feature_to_idx[cat][feat]
                    for cat, feat in zip(
                        self.feature_categories,
                        self.features[self.img_files[idx]],
                    )
                ]
                return (
                    source,
                    torch.tensor(cat, dtype=torch.long, device=self.device),
                    target,
                )
        elif self.use_txt_features == "embedded":
            return (
                source,
                self.bert_features[idx],
                self.clip_features[idx],
                target,
            )
        elif self.use_txt_features == "histogram":
            if self.pre_load_images:
                return (
                    source,
                    self.source_histograms[idx],
                    target,
                    self.target_histograms[idx],
                )
            else:
                source_hist = self.compute_histogram(source).to(self.device)
                target_hist = self.compute_histogram(target).to(self.device)
                return source, source_hist, target, target_hist
        else:
            raise ValueError(
                "Invalid value for use_txt_features. Must be False, 'one_hot', 'categorical', 'embedded', or 'histogram'."  # noqa: E501
            )

    def collate_fn(self, batch):
        if self.use_txt_features == "embedded":
            sources, bert_features, clip_features, targets = zip(*batch)
            return (
                torch.stack(sources),
                torch.stack(bert_features),
                torch.stack(clip_features),
                torch.stack(targets),
            )
        elif self.use_txt_features == "histogram":
            sources, source_hists, targets, target_hists = zip(*batch)
            return (
                torch.stack(sources),
                torch.stack(source_hists),
                torch.stack(targets),
                torch.stack(target_hists),
            )
        else:
            sources, features, targets = zip(*batch)
            return (
                torch.stack(sources),
                torch.stack(features) if features[0] is not None else None,
                torch.stack(targets),
            )
