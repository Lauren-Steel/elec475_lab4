import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def load_pretrained_fcn_resnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use the updated weights argument
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model = model.eval().to(device)

    print("Pre-trained FCN-ResNet50 model loaded successfully!")
    return model

