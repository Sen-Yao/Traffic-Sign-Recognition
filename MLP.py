import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib as plt

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, criterion, optimizer, train_loader, test_loader, epochs):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_loss = evaluate(model, criterion, test_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Test Loss: {test_loss/len(test_loader)}")

def evaluate(model, criterion, test_loader):
    model.eval()
    running_loss = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss

def test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 绘制混淆矩阵的灰度图
    conf = confusion_matrix(all_labels, all_preds)
    plt.imshow(conf, interpolation='nearest', cmap=plt.cm.viridis)
    plt.title('Confusion Matrix in Color')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(all_labels)))
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.clim(0, np.max(conf) / 2)  # 专注于非对角线上的较小值
    plt.show()

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def test_bagging(models, test_loader):
    all_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in models:
        model.eval()
        model_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                model_preds.extend(preds.cpu().numpy())
        all_preds.append(model_preds)

    # 通过投票机制组合预测结果
    all_preds = np.array(all_preds)
    final_preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(all_preds))).argmax(), axis=0, arr=all_preds)

    # 计算并打印Bagging模型的准确率
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, final_preds)
    print(f"Bagging Test Accuracy: {accuracy * 100:.2f}%")
