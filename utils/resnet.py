from keras.applications.resnet50 import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout


def resnet50_model():
    model = ResNet50(include_top=False, input_shape=(240,480,3))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='tanh'))
    model = Model(input=model.input, output=top_model(model.output))


    model.summary()
    return model
