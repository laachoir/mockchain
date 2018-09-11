import mockchain
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# parameters
ring_size = 8
num_conf_needed = 5
num_users = 20
num_blocks = 2000
train_test_ratio = 0.8

# generate mockchain
users = [mockchain.User("user_" + str(i),
		mining_power=np.random.random(),
		transaction_frequency=0.2,
		output_pick_strategy="old_first")
		for i in range(num_users)]
chain = mockchain.create_mockchain(users, num_blocks=num_blocks, minimum_ringsize=ring_size, confirmations_needed=num_conf_needed)
data = mockchain.get_mockchain_db(chain)

# pick relevant data points
num_real_data_points = len([row for row in data.iloc[:,3] if not row == None])
x_train = np.empty((num_real_data_points, ring_size), dtype=int)
y_train = np.empty((num_real_data_points, ring_size), dtype=int)
# TODO I don't yet understand how to access DataFrames in the way I want.
# TODO leaving out the coinbase transactions is so ugly right now.
k = 0
for i in range(len(data)):
	if not data.iloc[i][3] == None:
		x_train[k] = sorted(data.iloc[i][2])
		y_train[k] = np.array([int(j == data.iloc[i][3]) for j in x_train[k]]) # One-hot encode
		k += 1
train_test_threshold = int(len(x_train) * 0.8)
x_train, x_test = x_train[:train_test_threshold], x_train[:train_test_threshold]
y_train, y_test = y_train[:train_test_threshold], y_train[:train_test_threshold]
print("Guess constant index: " + str(sum(y_train)/sum(sum(y_train))))

# train model
model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=ring_size))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=ring_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=7, batch_size=32)

# test model
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
classes = model.predict(x_test, batch_size=64)
print(loss_and_metrics)
print(classes)
