# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:12:33 2019

@author: Shivani Sajjan
"""

import os
import glob
import cv2
import matplotlib.pylab as plt
import fnmatch
import numpy as np
import sklearn
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from scipy import ndimage
from collections import Counter 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.svm import SVC
print("MANIKANTA")

path=os.path.abspath(".")
data=os.path.join(path,'breast-histopathology-images')
image_path= glob.glob(os.path.join(data,"**/*.png"),recursive=True)

neg_pattern = '*class0.png'
pos_pattern = '*class1.png'
neg_images = fnmatch.filter(image_path, neg_pattern)
pos_images = fnmatch.filter(image_path, pos_pattern)

idc_neg_images=np.random.choice(neg_images,50000)
idc_pos_images=np.random.choice(pos_images,50000)
image_path_list=idc_neg_images.tolist()+idc_pos_images.tolist()
print("image_path_list")


print("HIIIIII")
labels=[]
y=[]
final_image=[]
final=[]
for img in image_path_list:
    image=cv2.imread(img)
    if image is not None:
        resized_images=cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
        #X1= np.array(resized_images,dtype='float32')
        final.append(resized_images)
        gray_image = cv2.cvtColor(resized_images, cv2.COLOR_BGR2GRAY)
        filtered = cv2.medianBlur(gray_image, 5)
        blurred_image = ndimage.gaussian_filter(filtered, sigma=3)
        mask=cv2.subtract(filtered,blurred_image)
        mask=mask*5
        filtered = cv2.add(filtered,mask)
        #ret, thresh = cv2.threshold(filtered,127,255,cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
        thresh = cv2.adaptiveThreshold(filtered,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        imarray=np.array(thresh,dtype=np.float64)
        n=np.fft.fft2(imarray)
        fshift = np.fft.fftshift(n)
        magnitude_spectrum = 20*np.log(np.abs(fshift)+1)
        mean_filtered = cv2.blur(magnitude_spectrum, (3,3))
        final_image.append(mean_filtered.flatten())
        if img in idc_neg_images:
            y.append(0)
        elif img in idc_pos_images:
            y.append(1) 
            
print("shivani")



X = np.array(final_image,dtype='float32')
X = X/255.0
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2 , random_state=0 , stratify=y)
print(Counter(Y_train))
X1= np.stack(final,axis=0)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, y, test_size=0.2 , random_state=0 , stratify=y)
print(Counter(Y1_train))
print(X1_train.shape)


# Look at confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
    
    
lr = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=10, multi_class='ovr', n_jobs=1,
          penalty='l1', random_state=None, solver='saga', tol=0.0001,
          verbose=0, warm_start=False)
graph=[]
name=[]
lr.fit(X_train,Y_train)
y_pred=lr.predict(X_test)
map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
confusion_mtx = confusion_matrix(Y_test, y_pred) 
plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 
plt.show()
tru=confusion_mtx[0][0]+confusion_mtx[1][1]
total=confusion_mtx[0][1]+confusion_mtx[0][0]+confusion_mtx[1][1]+confusion_mtx[1][0]
acc=tru/total
preci=confusion_mtx[0][0]+confusion_mtx[0][1]
print("acc:"+str(acc))
kfold = model_selection.KFold(n_splits=2)
accuracy = model_selection.cross_val_score(lr, X_test,Y_test, cv=kfold, scoring='accuracy')
#graph.append(accuracy)
mean = accuracy.mean() 
stdev = accuracy.std()
print('LogisticRegression - cross validation: %s (%s)' % (mean*100, stdev))

print("sgjzsdk")
dt=DecisionTreeClassifier(criterion = "entropy", random_state = 0,max_depth = 5,min_samples_leaf=10)
dt.fit(X_train, Y_train)
y_pred=dt.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
confusion_mtx = confusion_matrix(Y_test, y_pred) 
plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 
plt.show()
total=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
correct=cm[0][0]+cm[1][1]
percent=(correct/total)*100
print(percent)
print(y_pred.shape)


def MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)
        
        
        
print("HIIII")
batch_size = 128
num_classes = 2
epochs = 15
img_rows, img_cols = X1_train.shape[1],X1_train.shape[2]
input_shape = (img_rows, img_cols,3)
x_train = X1_train
y_train = Y1_train
x_test = X1_test
y_test = Y1_test  
y_test=np.array(y_test,dtype='uint32')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          verbose=1,
          epochs=epochs,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('\nKeras CNN #1A - accuracy:', score[1],'\n')
print("BYEEE")


model = SVC(gamma='auto')

model.fit(X_train, Y_train)
prediction = model.predict(X_test)
kfold = model_selection.KFold(n_splits=20)
accuracy = model_selection.cross_val_score(model, X_test,Y_test, cv=kfold, scoring='accuracy')
mean = accuracy.mean() 
stdev = accuracy.std()
print('\nSupport Vector Machine - Training set accuracy: %s (%s)' % (mean, stdev),"\n")
cm=confusion_matrix(Y_test,prediction)
map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
confusion_mtx = confusion_matrix(Y_test, prediction) 
plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 
plt.show()
total=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
correct=cm[0][0]+cm[1][1]
percent=(correct/total)*100
print(percent)



img_rows, img_cols = 50,50
input_shape = (img_rows, img_cols, 3)
batch_size = 128
num_classes = 2
epochs = 12
map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), padding='same')) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
datagen.fit(a)
model.fit_generator(datagen.flow(a,b, batch_size=32),
                    steps_per_epoch=len(a) / 32, epochs=epochs, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])
score = model.evaluate(c,d, verbose=0)
print('\nKeras CNN #3B - accuracy:', score[1],'\n')