# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

def build_siamese_model(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)

	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (10, 10), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(128, (7, 7), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	# below is the added by Jack Z on 21 Sept 2021
	x = Conv2D(128, (4, 4), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	# below is also the added by Jack Z on 21 Sept 2021
	x = Conv2D(256, (4, 4), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)

	# build the model
	model = Model(inputs, outputs)

	# return the model to the calling function
	return model