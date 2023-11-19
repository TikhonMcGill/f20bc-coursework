import pandas as pd

def prepare_data(test_size):
    dataset = pd.read_csv("data_banknote_authentication.txt") #Load the Dataset

    test_samples = int(test_size * len(dataset)) #Get the no. samples used for testing

    train_data = dataset.iloc[test_samples:] #Get the training data
    test_data = dataset.iloc[:test_samples] #Get the test data

    train_labels = train_data.iloc[:,-1] #Get the labels for the training data
    test_labels = test_data.iloc[:,-1] #Get the labels for the test data

    train_data.drop(columns=train_data.columns[len(train_data.columns)-1], inplace=True) #Drop the labels for the training data
    test_data.drop(columns=test_data.columns[len(test_data.columns)-1], inplace=True) #Drop the labels for the test data

    return test_data.to_numpy(), train_data.to_numpy(), train_labels.to_numpy(), test_labels.to_numpy()