
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob



IMAGE_SIZE = [224, 224]

train_path = r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\images\Train/'
valid_path = r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\images\Validation/'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


for layer in vgg.layers:
  layer.trainable = False
  


folders = glob(r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\images\Train/*')

  


x = Flatten()(vgg.output)

prediction = Dense(50, activation='relu')(x)


model = Model(inputs=vgg.input, outputs=prediction)


model.summary()


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)   #CNN concept


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\images\Train/',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\images\Validation/',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.save(r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\facefeatures_new_model.h5')
print("Model Trained Successfully")
