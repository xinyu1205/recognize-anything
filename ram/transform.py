from torchvision.transforms import Normalize, Compose, Resize, ToTensor


def get_transform(image_size=384):
    return Compose([
        lambda image: image.convert("RGB"),
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
