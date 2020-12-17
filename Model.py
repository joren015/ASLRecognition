import shutil
import uuid
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from os import makedirs, environ
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ASLDataset(Dataset):
    def __init__(self, csv_file):
        self.dataset = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = torch.load(self.dataset.iloc[idx]["FilePath"])
        label = self.dataset.iloc[idx]["LabelEncoded"]
        return img.float(), torch.tensor(label).long()
    

batch_size = 128
class ASLNet(nn.Module):
    def __init__(self):
        super(ASLNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(256 * 21 * 21, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 36)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout25 = nn.Dropout2d(p=0.25)
        self.dropout50 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.dropout25(x)
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.dropout25(x)
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.dropout25(x)
        x = x.view(-1, 256 * 21 * 21)
        x = F.relu(self.fc1(x))
        x = self.dropout50(x)
        x = F.relu(self.fc2(x))
        x = self.dropout50(x)
        x = self.fc3(x)
        return x


def main(train_path, test_path, val_path="", learning_rate=1e-3, batch_size=128, num_epochs=100, model_save_path="/content/models/misc"):
    model_name = "{}".format(uuid.uuid4())
    print(model_name)
    model_save_path_final = "{}/{}".format(model_save_path, model_name)
    makedirs(model_save_path_final)
    parameters = "learning_rate: {}\nbatch_size: {}\nnum_epochs: {}".format(learning_rate, batch_size, num_epochs)
    with open("{}/parameters.txt".format(model_save_path_final), 'x') as f:
      f.write(parameters)

    shutil.copyfile(train_path, "{}/train.csv".format(model_save_path_final))
    shutil.copyfile(test_path, "{}/test.csv".format(model_save_path_final))
    shutil.copyfile(val_path, "{}/validation.csv".format(model_save_path_final))
    print("Using device: {}".format(device))
    train_dataset = ASLDataset(train_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size
                              , drop_last=True, shuffle=True, num_workers=4)
    test_dataset = ASLDataset(test_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1
                              , drop_last=False, shuffle=False, num_workers=2)
    if val_path != "":
      use_validation = True
      val_dataset = ASLDataset(val_path)
      val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset)
                              , drop_last=False, shuffle=False, num_workers=4)
      best_val_accuracy = 0

    model = ASLNet().to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = None
    for t in range(num_epochs):
        if loss:
            print("Stats <Epoch: {} ::: Loss: {} ::: ValidationAcc: {}>".format((t-1), loss.item(), val_accuracy))
        
        for i, data in enumerate(train_loader, 0):
            X, y = data[0].to(device), data[1].to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if use_validation:
          correct = 0
          total = 0
          with torch.no_grad():
              for data in val_loader:
                  X, y = data[0].to(device), data[1].to(device)
                  y_pred = model(X)
                  _, predicted = torch.max(y_pred.data, 1)
                  total += y.shape[0]
                  correct += (predicted == y).sum().item()

          val_accuracy = 100 * (correct / total)
          if val_accuracy > best_val_accuracy:
            torch.save(model.state_dict(), '{}/best-model-parameters.pt'.format(model_save_path_final))
            best_val_accuracy = val_accuracy
    

    output = ""
    best_model = ASLNet().to(device)
    best_model.load_state_dict(torch.load('{}/best-model-parameters.pt'.format(model_save_path_final)))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            X, y = data[0].to(device), data[1].to(device)
            y_pred = best_model(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()

    output += "Validation: {}\n".format(100 * correct / total)
    print('Accuracy of the network on the validation images: %d %%' % 
          (100 * correct / total))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            X, y = data[0].to(device), data[1].to(device)
            y_pred = best_model(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()

    output += "Train: {}\n".format(100 * correct / total)
    print('Accuracy of the network on the train images: %d %%' % 
          (100 * correct / total))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            X, y = data[0].to(device), data[1].to(device)
            y_pred = best_model(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()

    output += "Test: {}".format(100 * correct / total)
    print('Accuracy of the network on the test images: %d %%' % 
          (100 * correct / total))

    with open("{}/results.txt".format(model_save_path_final), 'x') as f:
      f.write(output)

