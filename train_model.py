import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import transforms

data = torch.load('processed_data.pt')
targets = torch.load('processed_targets.pt')

# Data Augmentation
transform = transforms.Compose([
   transforms.RandomHorizontalFlip(),
   transforms.RandomRotation(10),
   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
   transforms.RandomVerticalFlip(p=0.5)
])

# Applying the transforms
# This assumes that 'data' is a tensor of images of the shape [N, C, H, W]
data = torch.stack([transform(img) for img in data])

# train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2)

# Splitting into training and temporary set (which will be divided into validation and test sets)
train_data, temp_data, train_targets, temp_targets = train_test_split(data, targets, test_size=0.3)

# Splitting the temporary set into validation and test sets
val_data, test_data, val_targets, test_targets = train_test_split(temp_data, temp_targets, test_size=0.5)

class SimpleCNN(nn.Module):
   def __init__(self, num_classes=5):
       super(SimpleCNN, self).__init__()
      
       self.conv_layers = nn.Sequential(
           nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
       )
      
       self.fc_layers = nn.Sequential(
           nn.Linear(32 * 56 * 56, 512),  # 56 is half of 224, assuming max pooling is applied twice
           nn.ReLU(),
           nn.Linear(512, num_classes)
       )
  
   def forward(self, x):
       x = self.conv_layers(x)
       x = x.view(x.size(0), -1)  # Flatten
       x = self.fc_layers(x)
       return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
num_epochs = 10

best_val_acc = 0  # for checkpointing
for epoch in range(num_epochs):
   # Training
   model.train()
   running_loss = 0.0
   for images, labels in zip(train_data, train_targets):
       optimizer.zero_grad()
       outputs = model(images.unsqueeze(0))  # unsqueeze to add batch dimension
       loss = criterion(outputs, labels.unsqueeze(0))
       loss.backward()
       optimizer.step()
       running_loss += loss.item()
   print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_data)}")
   
   # Validation loop
   model.eval()
   val_correct_predictions = 0
   val_loss = 0
   with torch.no_grad():
    for images, labels in zip(val_data, val_targets):
        outputs = model(images.unsqueeze(0))
        loss = criterion(outputs, labels.unsqueeze(0))
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        val_correct_predictions += (predicted == labels).sum().item()

   val_acc = 100 * val_correct_predictions / len(val_data)
   print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc}%, Validation Loss: {val_loss/len(val_data)}")

    # Model checkpointing
   if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), 'best_model.pth')
  
   # Evaluation (here we just print accuracy)
   '''
   model.eval()
   correct_predictions = 0
   with torch.no_grad():
       for images, labels in zip(test_data, test_targets):
           outputs = model(images.unsqueeze(0))
           _, predicted = torch.max(outputs, 1)
           correct_predictions += (predicted == labels).sum().item()
   print(f"Accuracy: {100 * correct_predictions / len(test_data)}%") '''

# After all epochs are completed, evaluate on test data
model.eval()
correct_predictions = 0
with torch.no_grad():
    for images, labels in zip(test_data, test_targets):
        outputs = model(images.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
print(f"Final Test Accuracy: {100 * correct_predictions / len(test_data)}%")


# save the model
torch.save(model.state_dict(), 'recycling_model.pth')