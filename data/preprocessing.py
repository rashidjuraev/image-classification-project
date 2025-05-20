import cv2
import numpy as np
from PIL import Image

class Preprocessing:
    """
    Class for image preprocessing methods
    """
    
    @staticmethod
    def resize_image(image, size):
        """
        Resize an image to the specified size
        
        Args:
            image: Image to resize
            size (tuple): Target size (width, height)
            
        Returns:
            PIL.Image: Resized image
        """
        if isinstance(image, np.ndarray):
            # Convert OpenCV image to PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        return image.resize(size, Image.BILINEAR)
    
    @staticmethod
    def center_crop(image, size):
        """
        Center crop an image to the specified size
        
        Args:
            image: Image to crop
            size (int or tuple): Target size
            
        Returns:
            PIL.Image: Cropped image
        """
        if isinstance(image, np.ndarray):
            # Convert OpenCV image to PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if isinstance(size, int):
            size = (size, size)
        
        width, height = image.size
        left = (width - size[0]) // 2
        top = (height - size[1]) // 2
        right = left + size[0]
        bottom = top + size[1]
        
        return image.crop((left, top, right, bottom))
    
    @staticmethod
    def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Normalize an image using the given mean and standard deviation
        
        Args:
            image (numpy.ndarray): Image to normalize
            mean (list): Mean values for each channel
            std (list): Standard deviation values for each channel
            
        Returns:
            numpy.ndarray: Normalized image
        """
        if isinstance(image, Image.Image):
            # Convert PIL to numpy
            image = np.array(image) / 255.0
        
        # Ensure image is in the format (H, W, C)
        if image.shape[0] == 3:  # If in format (C, H, W)
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize each channel
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    @staticmethod
    def preprocess_for_model(image, model_name):
        """
        Preprocess an image for a specific model
        
        Args:
            image: Image to preprocess
            model_name (str): Name of the model
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        if model_name in ['inception', 'xception']:
            # Inception and Xception models typically use 299x299 inputs
            size = 299
            image = Preprocessing.resize_image(image, (size, size))
            image = Preprocessing.normalize_image(image)
        elif model_name == 'mobilenet_v2':
            # MobileNetV2 typically uses 224x224 inputs
            size = 224
            image = Preprocessing.resize_image(image, (size, size))
            image = Preprocessing.normalize_image(image)
        elif model_name == 'resnet':
            # ResNet typically uses 224x224 inputs
            size = 224
            image = Preprocessing.resize_image(image, (size, size))
            image = Preprocessing.normalize_image(image)
        elif model_name == 'vgg':
            # VGG typically uses 224x224 inputs
            size = 224
            image = Preprocessing.resize_image(image, (size, size))
            image = Preprocessing.normalize_image(image)
        elif model_name == 'squeezenet':
            # SqueezeNet typically uses 224x224 inputs
            size = 224
            image = Preprocessing.resize_image(image, (size, size))
            image = Preprocessing.normalize_image(image)
        else:
            # Default preprocessing
            size = 224
            image = Preprocessing.resize_image(image, (size, size))
            image = Preprocessing.normalize_image(image)
        
        return image