import numpy as np

# Load the training data
train_data = np.load('SP_train.npy', allow_pickle=True)
print(type(train_data), train_data)
