import os
import tqdm
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

from css_model import CompactSemanticSegmentationModel
from compute_metrics import compute_miou, measure_inference_speed

def tensor_to_image(image_tensor):
    # normalize tensor to 0-1 range before converting to an image
    image_tensor = image_tensor - image_tensor.min()
    image_tensor = image_tensor / image_tensor.max()
    return (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')

def extract_features(model, x):
    output = model(x)
    return output['out']

def logits_to_class_mask(logits):
    return logits.argmax(dim=1)  # logits to class indices

def prepare_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.int64))
    ])
    dataset = VOCSegmentation(
        root='./data', 
        year='2012', 
        image_set='val', 
        download=True, 
        transform=transform, 
        target_transform=target_transform
    )

    # split validation set 80/20 to use for training/testing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader


def configure_models():
    teacher_model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT).eval()
    student_model = CompactSemanticSegmentationModel()
    student_model_path = 'results/models/custom_model.pth'
    if os.path.exists(student_model_path):
        student_model.load_state_dict(torch.load(student_model_path))
    return teacher_model, student_model


def train_distillation(train_loader, teacher_model, student_model, device):

    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ground truth loss
    softmax = nn.Softmax(dim=1)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    alpha, beta, temperature = 0.5, 0.5, 3  # weight for ground truth loss, teacher-student loss, temperature
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    total_start_time = time.time()

    # distillation training loop
    for epoch in range(10):  # 10 epochs 
        epoch_start_time = time.time() 
        student_model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass student model
            student_outputs = student_model(images)['out']
            student_features = extract_features(student_model, images)  # Extract intermediate features
            
            # forward pass  teacher model
            with torch.no_grad():
                teacher_outputs = teacher_model(images)['out']
                teacher_features = extract_features(teacher_model, images)

            student_mask = logits_to_class_mask(student_outputs) 
            teacher_mask = logits_to_class_mask(teacher_outputs) 

            # ground truth loss
            gt_loss = criterion(student_outputs, targets)

            teacher_probs = softmax(teacher_mask / temperature)
            student_probs = softmax(student_mask / temperature)

            distillation_loss = nn.KLDivLoss(reduction="batchmean")(torch.log(student_probs), teacher_probs) * (temperature ** 2)
            response_loss = alpha * gt_loss + beta * distillation_loss
            feature_loss = nn.MSELoss()(student_features, teacher_features)
            
            # backpropagation
            optimizer.zero_grad()
            response_loss.backward()
            optimizer.step()
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch Time: {epoch_duration:.2f} seconds")

        print(f"Epoch [{epoch+1}/10]\n" + 
            f"\tResponse-Based Loss: {response_loss.item()}" +
            f"\Feature-Based Loss: {feature_loss.item()}")
        
        final_model_path = "results/models/final_distilled_model.pth"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(student_model.state_dict(), final_model_path)
        print("Final distilled model saved to:", final_model_path)

    total_training_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    response_miou = evaluate_model(student_model, train_loader, device, criterion)
    feature_miou = evaluate_model(student_model, train_loader, device, criterion, feature_based=True)
    print(f"Response-based mIoU: {response_miou:.4f}")
    print(f"Feature-based mIoU: {feature_miou:.4f}")

    print("Knowledge Distillation Completed!")

def evaluate_model(model, test_loader, device, feature_based=False):
    model.eval()
    total_miou = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)['out']
            if feature_based:
                outputs = extract_features(model, images)  # Assuming feature extraction returns logits
            total_miou += compute_miou(outputs, targets)
    return total_miou / len(test_loader)

def main():
    train_loader, test_loader = prepare_data_loaders()
    teacher_model, student_model = configure_models()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    teacher_model.to(device)
    student_model.to(device)

    train_distillation(train_loader, teacher_model, student_model, device)

    # Evaluate both models
    response_based_miou = evaluate_model(student_model, test_loader, device, feature_based=False)
    feature_based_miou = evaluate_model(student_model, test_loader, device, feature_based=True)
    response_inference_speed = measure_inference_speed(student_model, test_loader, device)
    feature_inference_speed = measure_inference_speed(student_model, test_loader, device)  # Assuming same model

    print(f"Response-based mIoU: {response_based_miou:.4f}")
    print(f"Feature-based mIoU: {feature_based_miou:.4f}")
    print(f"Response-based Inference Speed: {response_inference_speed:.2f} ms/image")
    print(f"Feature-based Inference Speed: {feature_inference_speed:.2f} ms/image")



if __name__ == "__main__":
    main()