import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

class SalaryPredictor(nn.Module):
    def __init__(self):
        super(SalaryPredictor, self).__init__()
        # Define the architecture of your neural network here
        self.fc1 = nn.Linear(8,128)
        # Matrix of size 8 x 64
        # where 8 is the number of feature columns, 64 is the batch size
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define the forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)

def data_preprocessing(task_1a_dataframe):
    # Implement data preprocessing as described in the comments
    encoded_dataframe = task_1a_dataframe.copy()
    categorical_columns = ['Education', 'City', 'Gender', 'EverBenched','JoiningYear']
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        encoded_dataframe[col] = label_encoders[col].fit_transform(encoded_dataframe[col])
    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
    # Implement feature and target identification
    features = encoded_dataframe.drop(columns=['LeaveOrNot'])
    target = encoded_dataframe['LeaveOrNot']
    features_and_targets = [features, target]
    return features_and_targets

def load_as_tensors(features_and_targets):
    # Implement loading data as PyTorch tensors
    features, target = features_and_targets
    X_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(target.values, dtype=torch.float32)

    # Split the data into training and validation sets
    validation_split = 0.20
    num_val_samples = int(len(X_tensor) * validation_split)
    X_train_tensor = X_tensor[:-num_val_samples]
    y_train_tensor = y_tensor[:-num_val_samples]
    X_val_tensor = X_tensor[-num_val_samples:]
    y_val_tensor = y_tensor[-num_val_samples:]

    # Create DataLoader objects for training and validation
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    tensors_and_iterable_training_data = [X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader]
    return tensors_and_iterable_training_data

def model_loss_function():
    # Define your loss function here
    loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    return loss_function

def model_optimizer(model):
    # Define your optimizer here
    optimizer = torch.optim.Adam(params = model.parameters(), lr=0.0015)
    # optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            # lr=0.01)
    return optimizer

def model_number_of_epochs():
    # Define the number of training epochs
    number_of_epochs = 200  # You can adjust this as needed
    return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader = tensors_and_iterable_training_data

    for epoch in range(number_of_epochs):
        trained_model = model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = trained_model(inputs)
            loss = loss_function(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{number_of_epochs}, Loss: {total_loss / len(train_loader)}")

    return model

def validation_function(trained_model, tensors_and_iterable_training_data):
    X_val_tensor, y_val_tensor = tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3]
    trained_model.eval()
    with torch.inference_mode():
        outputs = trained_model(X_val_tensor)
        predicted = (outputs >= 0.5).float()
        correct = (predicted == y_val_tensor.unsqueeze(1)).sum().item()
        total = y_val_tensor.size(0)
        accuracy = correct / total
        return accuracy*100

if __name__ == "__main__":
    # Read the provided dataset CSV file
    task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

    # Data preprocessing
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # Identify features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # Load data as tensors
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    # Create the model
    # input_size = len(features_and_targets[0].columns)
    model = SalaryPredictor()

    # Define loss function, optimizer, and number of epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # Train the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)

    # Validate and obtain accuracy
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
    print(f"Accuracy on the validation set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
    # # Save the trained model
    # torch.save(trained_model.state_dict(), "task_1a_trained_model.pth")
