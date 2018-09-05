import mockchain
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# parameters
ring_size = 5
num_conf_needed = 5
num_users = 20
num_blocks = 200

# generate mockchain
users = []
for i in range(num_users):
	users += [mockchain.User("user_" + str(i), mining_power=np.random.random(), transaction_frequency=0.2, output_pick_strategy="old_first")]
chain = mockchain.create_mockchain(users, num_blocks=num_blocks, minimum_ringsize=ring_size, confirmations_needed=num_conf_needed)
data = mockchain.get_mockchain_db(chain)

# pick relevant data points
x_train = []
y_train = []
# TODO I don't yet understand how to access DataFrames in the way I want.
for i in range(len(data)):
	if not data.loc[i][3] == None:
		x_train += [data.loc[i][2]]
		y_train += [data.loc[i][3]]
print(y_train)


# train model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=ring_size))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=ring_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=5, batch_size=32)

# test model
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)