import tensorflow as tf
def gender_net():
	model=tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(1500, activation='relu', input_shape=(2622,)))
	model.add(tf.keras.layers.Dense(750, activation='relu'))
	model.add(tf.keras.layers.Dense(250, activation='relu'))
	model.add(tf.keras.layers.Dense(2))
	assert model.output_shape == (None, 2)
	return model

def age_net():
	model=tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(2000, activation='relu', input_shape=(2622,)))
	model.add(tf.keras.layers.Dense(1250, activation='relu'))
	model.add(tf.keras.layers.Dense(500, activation='relu'))
	model.add(tf.keras.layers.Dense(14))
	assert model.output_shape == (None, 14)
	return model
