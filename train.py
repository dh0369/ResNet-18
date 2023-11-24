from models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from plain_models import Plain18, Plain34
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                        std=[0.2470, 0.2435, 0.2616])])
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #                     std=[0.229, 0.224, 0.225])])
                                    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    #                     std=[0.5, 0.5, 0.5])])


    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root='./data', train=False, 
                               download=True, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

def main():

    batch_size = 128
    epochs = 150
    learning_rate = 1E-4
    train_loader, test_loader, classes = dataset(batch_size)

    # model = Plain18()
    # model = Plain34()
    # model = ResNet18()
    # model = ResNet34()
    # model = ResNet50()
    # model = ResNet101()
    model = ResNet152()
    print(model)

    model = model.to(device)

    summary(model, input_size=(3, 32, 32))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss = running_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            print('Epoch: %03d/%03d | Train: %.2f%% | Loss: %.3f' % (
                epoch + 1, epochs, compute_accuracy(model, train_loader), running_loss))
            print('               | Test Accuracy: %.2f%%' % (
                compute_accuracy(model, test_loader)))
            
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    sample_labels = sample_labels.to(device)

    sample_outputs = model(sample_images)
    _, sample_predicted = torch.max(sample_outputs, 1)

    num_samples = 5

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        sample_img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
        sample_img = 0.5 * sample_img + 0.5
        axes[i].imshow(sample_img)
        axes[i].set_title(f'Predicted: {classes[sample_predicted[i]]}\nActual: {classes[sample_labels[i]]}')

    plt.show()

def compute_accuracy(model, data_loader):
    correct, total = 0, 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted_labels == labels).sum()

    return correct.float()/total * 100
    
if __name__ == '__main__':
    main()