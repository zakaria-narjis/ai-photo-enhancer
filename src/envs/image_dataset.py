
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2.functional as F
import random
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as F
import random
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
from tqdm import tqdm

class FiveKDataset(Dataset):
    def __init__(self, image_size, mode="train", resize=True, 
                 augment_data=False, use_txt_features=False,device='cuda'):
        current_dir = os.getcwd()
        dataset_dir = os.path.join(current_dir, "..", "dataset")
        self.IMGS_PATH = os.path.join(dataset_dir, f"FiveK/{mode}")
        self.FEATURES_PATH = os.path.join(dataset_dir, "processed_categories.txt")
        
        self.resize = resize
        self.image_size = image_size
        self.augment_data = augment_data
        self.use_txt_features = use_txt_features
        self.device = device
        self.img_files = []
        self.features = {}
        self.feature_categories = ["Location", "Time", "Light", "Subject"]
        
        # Load semantic features from processed_categories.txt
        with open(self.FEATURES_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                img_name = parts[0]
                img_features = parts[1:]
                self.features[img_name] = img_features
        
        # Load image files
        self.img_files = [f for f in os.listdir(os.path.join(self.IMGS_PATH, 'input'))]
        
        # Prepare MultiLabelBinarizer
        all_features = [self.features[img] for img in self.img_files]
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(all_features)
        
        # Create encoding dictionaries for categorical approach
        unique_features = {cat: set(feat[i] for feat in all_features) 
                           for i, cat in enumerate(self.feature_categories)}
        self.feature_to_idx = {
            cat: {feat: idx for idx, feat in enumerate(sorted(features))}
            for cat, features in unique_features.items()
        }
        
        self.precomputed_bert_features = {}
        self.precomputed_clip_features = {}
        
        if self.use_txt_features == "embedded":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
            self.bert_model.to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            self.clip_model.to(self.device)
            self.precompute_features()

    def precompute_features(self):
        print("Precomputing BERT and CLIP features...")
        for img_name in tqdm(self.img_files):
            # Precompute BERT features
            feature_text = " ".join(self.features[img_name])
            inputs = self.tokenizer(feature_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():              
                outputs = self.bert_model(**inputs.to(self.device))
            bert_features = outputs.last_hidden_state[:, 0, :].squeeze(0)  # Shape: (768,)
            self.precomputed_bert_features[img_name] = bert_features.cpu()

            # Precompute CLIP features
            image_path = os.path.join(self.IMGS_PATH, 'input', img_name)
            image = read_image(image_path)
            clip_inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():           
                clip_features = self.clip_model.get_image_features(**clip_inputs.to(self.device))

            self.precomputed_clip_features[img_name] = clip_features.squeeze(0).cpu()  # Shape: (512,)
        del self.bert_model
        del self.tokenizer    
        del self.clip_model
        del self.clip_processor  
        print("BERT and CLIP features precomputed and stored.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        source = read_image(os.path.join(self.IMGS_PATH, 'input', img_name))
        target = read_image(os.path.join(self.IMGS_PATH, 'target', img_name))

        if self.resize:
            source = F.resize(source, (self.image_size, self.image_size), interpolation=F.InterpolationMode.BICUBIC)
            target = F.resize(target, (self.image_size, self.image_size), interpolation=F.InterpolationMode.BICUBIC)

        if self.augment_data:
            if random.random() > 0.5:
                source = F.hflip(source)
                target = F.hflip(target)
            if random.random() > 0.5:
                source = F.vflip(source)
                target = F.vflip(target)

        if not self.use_txt_features:
            return source, target

        elif self.use_txt_features == "one_hot":
            one_hot_features = self.mlb.transform([self.features[img_name]])[0]
            one_hot_features = torch.tensor(one_hot_features, dtype=torch.float32)
            return source, one_hot_features, target

        elif self.use_txt_features == "categorical":
            cat_features = [self.feature_to_idx[cat][feat] for cat, feat in zip(self.feature_categories, self.features[img_name])]
            cat_features = torch.tensor(cat_features, dtype=torch.long)
            return source, cat_features, target

        elif self.use_txt_features == "embedded":
            bert_features = self.precomputed_bert_features[img_name]
            clip_features = self.precomputed_clip_features[img_name]
            return source, bert_features, clip_features, target

        else:
            raise ValueError("Invalid value for use_txt_features. Must be False, 'one_hot', 'categorical', or 'embedded'.")

    def collate_fn(self, batch):
        if self.use_txt_features == "embedded":
            sources, bert_features, clip_features, targets = zip(*batch)
            sources = torch.stack(sources)
            bert_features = torch.stack(bert_features)
            clip_features = torch.stack(clip_features)
            targets = torch.stack(targets)
            return sources, bert_features, clip_features, targets
        else:
            sources, features, targets = zip(*batch)
            sources = torch.stack(sources)
            targets = torch.stack(targets)
            if self.use_txt_features in ["one_hot", "categorical"]:
                features = torch.stack(features)
            else:
                features = None
            return sources, features, targets
    
