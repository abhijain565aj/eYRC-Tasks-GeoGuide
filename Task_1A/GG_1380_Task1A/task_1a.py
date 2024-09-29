'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_1380
# Author List:		[Abhi Jain, Sameer Arvind Patil, Anushka Singhal, Arya Suwalka]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas
import torch
import numpy 
###################### Additional Imports ####################
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################





##############################################################

def data_preprocessing(task_1a_dataframe):
	''' 
	Purpose:
	---
	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that 
	there are features in the csv file whose values are textual (eg: Industry, 
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function 
	should return encoded dataframe in which all the textual features are 
	numerically labeled.
	
	Input Arguments:
	---
	`task_1a_dataframe`: [Dataframe]
						Pandas dataframe read from the provided dataset 	
	
	Returns:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''
	encoded_dataframe=task_1a_dataframe.copy()
    # Identify the columns with textual values that need encoding
	categorical_columns = ['Education','City','Gender','EverBenched','JoiningYear']
	# Initialize a LabelEncoder for each categorical column
	label_encoders = {}
	for col in categorical_columns:
		label_encoders[col]=LabelEncoder()
		encoded_dataframe[col] = label_encoders[col].fit_transform(encoded_dataframe[col])
	return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
	'''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label

	Input Arguments:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	
	Returns:
	---
	`features_and_targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label

	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''

	# Defining features and target labels here
	# target column = LeaveOrNot
	features = encoded_dataframe.drop(columns=['LeaveOrNot'])
	target = encoded_dataframe['LeaveOrNot']

	features_and_targets = [features, target]

	return features_and_targets

def load_as_tensors(features_and_targets):

	''' 
	Purpose:
	---
	This function aims at loading your data (both training and validation)
	as PyTorch tensors. Here you will have to split the dataset for training 
	and validation, and then load them as as tensors. 
	Training of the model requires iterating over the training tensors. 
	Hence the training sensors need to be converted to iterable dataset
	object.
	
	Input Arguments:
	---
	`features_and targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label
	
	Returns:
	---
	`tensors_and_iterable_training_data` : [ list ]
											Items:
											[0]: X_train_tensor: Training features loaded into Pytorch array
											[1]: X_test_tensor: Feature tensors in validation data
											[2]: y_train_tensor: Training labels as Pytorch tensor
											[3]: y_test_tensor: Target labels as tensor in validation data
											[4]: Iterable dataset object and iterating over it in 
												batches, which are then fed into the model for processing

	Example call:
	---
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	'''
	# Extract features and target from the input list
	features, target = features_and_targets
    # Convert Pandas DataFrames to PyTorch tensors
	# Capital letters used to denote Matrices and more than 2 dim tensors while small letters used to denote Scalars and Vectors
	X_tensor = torch.tensor(features.values, dtype=torch.float32)
	y_tensor = torch.tensor(target.values, dtype=torch.float32)

	#factor by which we need to get the split
	validation_split = 0.20
	# Calculate the number of samples in the validation set
	num_val_samples = int(len(X_tensor) * validation_split)
	# Split the data into training and validation sets
	X_train_tensor = X_tensor[:-num_val_samples]
	y_train_tensor = y_tensor[:-num_val_samples]
	X_val_tensor = X_tensor[-num_val_samples:]
	y_val_tensor = y_tensor[-num_val_samples:]
	
	# Create a TensorDataset for training and validation
	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
	
	# batch size is amount of data to be taken at once for training and other purposes
	batch_size = 64
	
	# Create DataLoader objects for training and validation
	# shuffle: Setting shuffle=True shuffles the data for each epoch (an epoch is one pass through the entire dataset) to introduce randomness into the training process.
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	
	# Return a list containing the tensors and the iterable training data
	tensors_and_iterable_training_data = [X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader]

	return tensors_and_iterable_training_data

import torch.nn as nn

class Salary_Predictor(nn.Module):
	'''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers. 
	It defines the sequence of operations that transform the input data into 
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and 
	the output is the prediction of the model based on the input data.
	
	Returns:
	---
	`predicted_output` : Predicted output for the given input data
	'''
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		'''
		Define the type and number of layers
		Hidden Layers = 3 (128, 64, 64 neurons respectively)
		'''
		No_Of_Features_Col = 8
		self.fc1 = nn.Linear(No_Of_Features_Col,128)
		#Matrix of sixe 8 x 128
		self.fc2 = nn.Linear(128,64)
		self.fc3 = nn.Linear(64,64)
		self.fc4 = nn.Linear(64,1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		'''
		Define the activation functions
		'''
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.relu(self.fc3(x))
		x = self.fc4(x)
		predicted_output = self.sigmoid(x)

		return predicted_output

def model_loss_function():
	'''
	Purpose:
	---
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values 
	in training data.
	
	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	loss_function = nn.BCELoss()
	return loss_function

def model_optimizer(model):
	'''
	Purpose:
	---
	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	Input Arguments:
	---
	`model`: An object of the 'Salary_Predictor' class

	Returns:
	---
	`optimizer`: Pre-defined optimizer from Pytorch

	Example call:
	---
	optimizer = model_optimizer(model)
	'''
	optimizer = torch.optim.Adam(params = model.parameters(),lr=0.0015)
	#params = parameters of target model to opimize
	#lr = learning rate
	return optimizer

def model_number_of_epochs():
	'''
	Purpose:
	---
	To define the number of epochs for training the model

	Input Arguments:
	---
	None

	Returns:
	---
	`number_of_epochs`: [integer value]

	Example call:
	---
	number_of_epochs = model_number_of_epochs()
	'''
	number_of_epochs =200 #can be edited
	return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	'''
	Purpose:
	---
	All the required parameters for training are passed to this function.

	Input Arguments:
	---
	1. `model`: An object of the 'Salary_Predictor' class
	2. `number_of_epochs`: For training the model
	3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											and iterable dataset object of training tensors
	4. `loss_function`: Loss function defined for the model
	5. `optimizer`: Optimizer defined for the model

	Returns:
	---
	trained_model

	Example call:
	---
	trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

	'''	
	# X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, 
	train_loader = tensors_and_iterable_training_data[4]

	for epoch in range(number_of_epochs):
		#put the model in training mode
		trained_model = model.train()
		total_loss = 0.0
		for inputs, y_targets in train_loader:
			#Zero the optimiser gradients as they accumulate by default
			optimizer.zero_grad()

			#The predicted outputs by running the forward method of class
			y_predicted = trained_model(inputs)

			#Calculating the loss
			loss = loss_function(y_predicted, y_targets.unsqueeze(1))

			#Performing back_propogation on loss
			loss.backward()

			#Progress the optimizer for calculating the gradients
			optimizer.step()

			total_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{number_of_epochs}, Loss: {total_loss / len(train_loader)}")


	return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
	'''
	Purpose:
	---
	This function will utilise the trained model to do predictions on the
	validation dataset. This will enable us to understand the accuracy of
	the model.

	Input Arguments:
	---
	1. `trained_model`: Returned from the training function
	2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											and iterable dataset object of training tensors

	Returns:
	---
	model_accuracy: Accuracy on the validation dataset

	Example call:
	---
	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

	'''	
	
	X_val_tensor = tensors_and_iterable_training_data[1]
	y_val_tensor = tensors_and_iterable_training_data[3]
	
	#Set the model in evalutation mode
	trained_model.eval()
	
	with torch.inference_mode():
		#forward pass on trained_model
		y_pred = trained_model(X_val_tensor)
		# converts the model outputs to binary predictions. Values greater than 0.5 are considered as class 1, and values less than or equal to 0.5 are considered as class 0. The .float() method ensures that the predictions are in a floating-point format
		y_pred_binary = (y_pred>=0.5).float()

		#Calculating number of correct predictions
		correct = (y_pred_binary == y_val_tensor.unsqueeze(1)).sum().item()

		total = y_val_tensor.size(0)
		
		model_accuracy = (correct/total)

	return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")