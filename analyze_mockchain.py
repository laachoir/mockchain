import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# parameters
ring_size = 5
with open("gen_data.csv") as data_input:
	data = pd.read_csv(data_input, sep='\t', index_col=0)
	print(data[:30])

# parse csv_file
print(data[2]) # TODO this breaks for some reason. Understand better how DataFrames work

# train model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=ring_size))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=ring_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=5, batch_size=32)

# test model
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)