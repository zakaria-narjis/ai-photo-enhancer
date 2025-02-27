from .image_dataset import FiveKDataset
from torch.utils.data import DataLoader


class PhotoEnhancement:
    """
    Encode dataset images
    output : torch.Tensors of encoded and raw source/target images (3,H,W)
    """

    def __init__(
        self,
        image_size,
        mode="train",
        resize=True,
        augment_data=False,
        use_txt_features=False,
        pre_load_images=True,
        device="cuda:0",
    ) -> None:
        self.image_size = image_size
        self.mode = mode
        self.resize = resize
        self.augment_data = augment_data
        self.use_txt_features = use_txt_features
        self.pre_load_images = pre_load_images
        self.device = device

    def generate_dataset(self):
        return FiveKDataset(
            image_size=self.image_size,
            mode=self.mode,
            resize=self.resize,
            augment_data=self.augment_data,
            use_txt_features=self.use_txt_features,
            device=self.device,
            pre_load_images=self.pre_load_images,
        )


def create_dataloaders(
    batch_size,
    image_size,
    use_txt_features=False,
    train=True,
    augment_data=False,
    shuffle=True,
    resize=True,
    pre_encoding_device="cuda",
    pre_load_images=True,
):
    if train:
        train_dataset = PhotoEnhancement(
            image_size,
            mode="train",
            resize=resize,
            augment_data=augment_data,
            use_txt_features=use_txt_features,
            device=pre_encoding_device,
            pre_load_images=pre_load_images,
        )
        train_dataset = train_dataset.generate_dataset()
        dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
    else:
        test_dataset = PhotoEnhancement(
            image_size,
            mode="test",
            resize=resize,
            augment_data=augment_data,
            use_txt_features=use_txt_features,
            device=pre_encoding_device,
            pre_load_images=pre_load_images,
        )
        test_dataset = test_dataset.generate_dataset()
        dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle
        )

    return dataloader
