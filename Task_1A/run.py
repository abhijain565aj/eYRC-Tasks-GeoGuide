####################### IMPORT MODULES #######################
import pandas
import torch
import numpy 
###################### Additional Imports ####################
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
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
	encoded_dataframe['Age'] = encoded_dataframe['Age']/10
	for col in categorical_columns:
		label_encoders[col]=LabelEncoder()
		encoded_dataframe[col] = label_encoders[col].fit_transform(encoded_dataframe[col])
	return encoded_dataframe


	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
encoded_dataframe = data_preprocessing(task_1a_dataframe)
print(encoded_dataframe['Age'])