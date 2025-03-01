#done in blocks because I did this in jupyter notebook, as I always do with machine learning

#code I was given
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "./data"

# Display info about the battery cell data which ID = 0

cell_idx = 0

cell_data_x = np.load(os.path.join(DATA_PATH, f"{cell_idx}_x.npy"))
cell_data_y = np.load(os.path.join(DATA_PATH, f"{cell_idx}_y.npy"))

print(f"Cell ID: {cell_idx}")
print(f"X shape: {cell_data_x.shape}")
print(f"y shape: {cell_data_y.shape}")

cycles = np.arange(len(cell_data_x)) + 1
plt.plot(cycles, cell_data_y.reshape(-1), label=f"Cell-0")
plt.xlabel("Cycles")
plt.ylabel("SoH (%)")
plt.legend()
plt.grid()
plt.show()

#Code I made myself

import tensorflow as tf
import pandas as pd
from random import randint #to choose the cell I would predict
print(cell_data_x[0])

#converting the raw data into a pandas dataframe
my_columns = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"]
x_data = []
y_data = []
predic = randint(0,47)
for i in range(48):
    if i != predic:
        x = np.load(os.path.join(DATA_PATH, f"{i}_x.npy"))
        y = np.load(os.path.join(DATA_PATH, f"{i}_y.npy"))
        for j in range(len(x)):
            x_data.append([])
            for k in range(16):
                x_data[len(x_data)-1].append(x[j][k])
            y_data.append(y[j][0])
    dataset = pd.DataFrame(data=x_data,columns=my_columns)
    dataset["Label"] = y_data

#to see correlations
import seaborn as sns
sns.heatmap(dataset.corr(), cmap = "crest")

#to see data structure so I could decide best form of normalisation
print(trainset.dtypes)
for i in range(1, 17):
    trainset.hist(str(i))

#normalisation and getting rid of useless features
dataset = dataset.drop(columns = ["6","7","9","11","12","13"])
def z_score(value):
    global mean
    global std
    return (value-mean)/std
for i in dataset.columns:
    if i != "Label":
        mean = np.mean(dataset[i])
        std = np.std(dataset[i])
        dataset[i] = dataset[i].apply(z_score)

#split into train and test
dataset = dataset.reindex(np.random.permutation(dataset.index))
trainset = dataset.iloc[:40000]
testset = dataset.iloc[40000:]

#dnn because ffn was abysmal, there are input prepping
input = {
    "1": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "2": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "3": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "4": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "5": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "8": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "10": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "14": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "15": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "16": tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    }
normalized_inputs = {}
for key, value in input.items():
    temp = tf.keras.layers.Normalization(axis = None) #I mean I already normalised but it works so...
    temp.adapt(trainset[key])
    temp = temp(input.get(key))
    normalized_inputs[key] = temp

#making the model
input_layer = tf.keras.layers.Concatenate()([value for key, value in normalized_inputs.items()])
dense_output = tf.keras.layers.Dense(units=400, activation = 'relu', name = 'hidden_dense_layer_1')(input_layer)
dense_output = tf.keras.layers.Dense(units=200, activation = 'relu', name = 'hidden_Dense_layer_2')(dense_output)
dense_output = tf.keras.layers.Dense(units=100, activation = 'relu', name = 'dense_output')(dense_output)
outputs = {"dense_output": dense_output}
dnn_model = tf.keras.Model(inputs = input, outputs = outputs)
dnn_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    loss = tf.keras.losses.mean_squared_error
)

#running and plotting mse and then evaluating
xtrain, ytrain = trainset.drop(columns = ["Label"]), trainset["Label"]
xtest, ytest = testset.drop(columns = ["Label"]), testset["Label"]
xtrain = {name:np.array(value) for name, value in xtrain.items()}
xtest = {name:np.array(value) for name, value in xtest.items()}
history = dnn_model.fit(x=xtrain,y=ytrain, batch_size = 5000, epochs = 300, validation_split = 0.25)
epochs = history.epoch
hist = pd.DataFrame(history.history)
mse = hist["loss"]
plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(epochs, mse, label="Training mse")
plt.plot(epochs, history.history["val_loss"], label = "Validation mse")
mergedmse = mse.tolist() + history.history["val_loss"]
plt.ylim([min(mergedmse)*0.97, max(mergedmse)*1.03])
plt.legend()
plt.show()
dnn_model.evaluate(x = xtest, y = ytest, return_dict = True)

#time to predict the final cell
x_data = []
y_data = []
x = np.load(os.path.join(DATA_PATH, f"{predic}_x.npy"))
y = np.load(os.path.join(DATA_PATH, f"{predic}_y.npy"))
for j in range(len(x)):
    x_data.append([])
    for k in range(16):
        x_data[len(x_data)-1].append(x[j][k])
    y_data.append(y[j][0])
dataset = pd.DataFrame(data=x_data,columns=my_columns)
dataset = dataset.drop(columns=["6","7","9","11","12","13"])
def z_score(value):
    global mean
    global std
    return (value-mean)/std
for i in dataset.columns:
    if i != "Label":
        mean = np.mean(dataset[i])
        std = np.std(dataset[i])
        dataset[i] = dataset[i].apply(z_score)
        mean = np.mean(dataset[i])
        std = np.std(dataset[i])
        dataset[i] = dataset[i].apply(z_score)
y_pred = dnn_model.predict({name:np.array(value) for name, value in dataset.items()})

#prediction comes as a weird 2d array dictionary so gotta convert that to 1d array
y_pred = y_pred["dense_output"]
alr = []
for i in range(len(y_pred)):
    alr.append(y_pred[i][0])
for i in range(2,len(alr)-2):
    moving_median = sorted([alr[i-2],alr[i-1],alr[i],alr[i+1],alr[i+2]])
    if abs(alr[i]-moving_median[2]) > 1.5:
        alr[i] = moving_median[2]

#plot predictions against actual values
cycles = np.arange(len(x_data)) + 1
plt.figure()
plt.xlabel("cycle")
plt.ylabel("SoH (%)")
plt.plot(cycles, alr, label="predicted")
plt.plot(cycles, y_data, label = "actual")
plt.ylim([0, 100])
plt.legend()
plt.show()
