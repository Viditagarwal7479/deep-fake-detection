import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def validate(loader, model):
    correct = 0
    total = 0
    model.eval()
    # No need to compute gradients during validation of model,  required_grad = False for all the parameters
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).max(1)[1]
            correct += sum(labels == outputs)
            total += len(labels)
    return 100 * correct / total, correct, total


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.inception_v3(pretrained=True)
model.dropout = nn.Dropout(p=0.5, inplace=True)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
batch_size = 16
epoch = 20
train_data = datasets.ImageFolder(root='images/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.ImageFolder(root='images/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

# Training the model
best_accuracy = 0
for e in range(epoch):
    losses = []
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits  # As in training auxillary loss is also returned
        loss = loss_function(outputs, labels)
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print("Epoch : {} , Loss : {}".format(e, sum(losses) / len(losses)))
    accuracy, _, _ = validate(test_loader, model)
    # Saving the model parameters after every epoch
    if accuracy > best_accuracy:
        torch.save(model.state_dict(), "InceptionV3.pt")

model.load_state_dict(torch.load("InceptionV3.pt"))
# At the end of training the model on the validation data.
epoch = 5
# Reducing the learning rate so that the model converges properly
for e in range(epoch):
    losses = []
    model.train()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).logits
        loss = loss_function(outputs, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print("Epoch : {} , Loss : {}".format(e, sum(losses) / len(losses)))

torch.save(model.state_dict(), "InceptionV3.pt")
