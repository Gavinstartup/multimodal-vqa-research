import torch
import torch.nn as nn
import logging
from PIL import Image
from config import VIT_MODEL_NAME, VIT_PRETRAINED, DEVICE
import open_clip

_vit_model = None

def open_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        logging.error(f"Opening images failed!: {str(e)}")
        raise
    # try to open the images and make sure the images are converted to RGB
    
def load_vit_and_preprocessor():
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=VIT_MODEL_NAME,
            pretrained=VIT_PRETRAINED,
        )
        model = model.to(DEVICE)
        model.eval()
        return model, preprocess
    except Exception as e:
        logging.error(f"Loading ViT Model failed! : {str(e)}")
        raise

def preprocess_image(image, preprocess_func):
    try:
        image_tensor = preprocess_func(image).unsqueeze(0).to(DEVICE)  # add batch dimension and move to device
        return image_tensor # add batch dimension and move to device finally get the tensor. The shape is like [1, 3, 224, 224]
    except Exception as e:
        logging.error(f"Preprocessing image failed! : {str(e)}")
        raise

    """
    这个文件的主要作用是：
    1.先打开图片
    2.使用huggingface的open_clip库加载vit模型
    3.加载vit模型和预处理函数来处理图像
    4.最后返回处理后的图像特征
    """