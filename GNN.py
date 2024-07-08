import torch
import torch.nn.functional as F
import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNClassifier:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, epochs=200):
        self.model = GCN(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.model.train()
        for epoch in range(self.epochs):
            for data in X_train:  # Assuming X_train is a DataLoader
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        predictions = []
        for data in X_test:  # Assuming X_test is a DataLoader
            out = self.model(data)
            pred = out.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
        return predictions


def create_dataloader(X, y_input):
    data_list = []
    for i in range(len(X)):
        x = torch.tensor(X[i], dtype=torch.float).view(-1, 1)  # reshape to (num_nodes, num_node_features)
        edge_index = torch.tensor([[i, j] for i in range(x.size(0)) for j in range(x.size(0))], dtype=torch.long).t().contiguous()  # fully connected graph
        y = torch.tensor([y_input[i]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return DataLoader(data_list, batch_size=32, shuffle=True)


def create_gnn_data(X, y, num_classes):
    edge_index = torch.tensor([[i, j] for i in range(len(X)) for j in range(len(X)) if i != j], dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def create_dataloader(data, batch_size=32):
    return DataLoader([data], batch_size=batch_size)

def create_gnn_data_with_progress(X, y, num_classes):
    data_list = []
    for i in tqdm(range(len(X)), desc="Converting data"):
        data = create_gnn_data(X[i:i+1], y[i:i+1], num_classes)
        data_list.append(data)
    return data_list