import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

# MLP module
class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size * input_size * 3, int(input_size/2))
        self.bn1 = torch.nn.BatchNorm1d(int(input_size/2))
        self.fc2 = torch.nn.Linear(int(input_size/2), int(input_size/4))
        self.bn2 = torch.nn.BatchNorm1d(int(input_size/4))
        self.fc3 = torch.nn.Linear(int(input_size/4), int(input_size/8))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Feature attention analysis module
class FeatureExtractor():
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.feature_maps = None
        self.gradients = None
        self.hook = None

    def hook_fn(self, module, input, output):
        self.feature_maps = output.detach()

    def hook_grad_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def register_hooks(self):
        layer = dict([*self.model.named_modules()])[self.layer_name]
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.grad_hook = layer.register_backward_hook(self.hook_grad_fn)

    def remove_hooks(self):
        self.hook.remove()
        self.grad_hook.remove()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def attention_analysis():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(3):
        os.makedirs('1000yen_result', exist_ok=True)
        os.makedirs('5000yen_result', exist_ok=True)
        os.makedirs('10000yen_result', exist_ok=True)
        for j in range(1, 21):
            if i == 1:
                image_path = f'bill_dataset/test/1000yen/{j}.jpg'
            elif i == 2:
                image_path = f'bill_dataset/test/5000yen/{j}.jpg'
            else:
                image_path = f'bill_dataset/test/10000yen/{j}.jpg'
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            feature_extractor = FeatureExtractor(model, 'layer4')
            feature_extractor.register_hooks()
            image_tensor.requires_grad_()
            output = model(image_tensor)
            output_idx = output.argmax(dim=1)
            output_max = output[0, output_idx]
            output_max.backward()
            feature_maps = feature_extractor.feature_maps
            gradients = feature_extractor.gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            weighted_feature_map = torch.sum(weights * feature_maps, dim=1)

            heatmap = weighted_feature_map.squeeze().cpu().detach().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
            heatmap = np.uint8(255 * (1 - heatmap / np.max(heatmap)))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            image_np = np.array(image)
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            image_gray_3ch = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            superimposed_img = heatmap*0.4 + image_gray_3ch*0.4

            plt.imshow(superimposed_img.astype('uint8'))
            plt.axis('off')
            if i == 1:
                plt.savefig(f'1000yen_result/output{j}.png')
            elif i == 2:
                plt.savefig(f'5000yen_result/output{j}.png')
            else:
                plt.savefig(f'10000yen_result/output{j}.png')
            feature_extractor.remove_hooks()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    # data transforms
    input_size = 256
    train_transform = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop((input_size, input_size), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # set and load dataset
    train_dataset = datasets.ImageFolder(root='bill_dataset/train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root='bill_dataset/test', transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

    # set model
    model = MLP(input_size).to(device)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    # intialize parameters for train
    epochs = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_losses, test_losses = [], []
    best_test_loss = 10000

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100 * correct / total

        # record best model based on test loss
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            best_accuracy = accuracy
            best_preds = all_preds[:]
            best_labels = all_labels[:]
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    # output test result in terminal 
    print("\nBest Test Results:")
    print(f"Epoch: {best_epoch}, Test Loss: {best_test_loss:.4f}, Test Accuracy: {best_accuracy:.2f}%")

    # plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='o')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('loss_curve.png')

    # compute confusion metrix
    confusion_matrix = torch.zeros(3, 3)
    for t, p in zip(best_labels, best_preds):
        confusion_matrix[t, p] += 1
    class_names = ['1000yen', '5000yen', '10000yen']
    class_order = [train_dataset.classes.index(cls) for cls in class_names]
    confusion_matrix = confusion_matrix[class_order, :][:, class_order]
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Epoch {best_epoch})")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, int(confusion_matrix[i, j].item()), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_best_epoch.png')

    # save model
    torch.save(model.state_dict(), "mlp.pth")


if __name__ == "__main__":
    main()
