import torch
from torchvision.models.segmentation import fcn_resnet50

def load_pretrained_teacher(device):
    # Load FCN ResNet50 pre-trained on COCO/VOC dataset
    teacher_model = fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
    teacher_model.eval()  # Set to evaluation mode
    teacher_model.to(device)  # Move to device
    return teacher_model