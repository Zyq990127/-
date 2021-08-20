from keras.models import load_model
import data_prosess
import os
import cv2
import random
import numpy as np

bath_path = 'train_data'
X_test, Y_test = data_prosess.load_dataset(bath_path)
model = load_model('resnet_224.h5')

preds = model.evaluate(X_test, Y_test)
print('Loss : ' + str(preds[0]))
print('accuracy : ' + str(preds[1]))
'''
img_path = '0 (12).jpg'
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (160, 160))
x = image/255
x = np.expand_dims(x, axis=-1)
x = np.expand_dims(x, axis=0)
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))
array_l = np.array(model.predict(x))
a = "%d"%np.argmax(array_l)
#cv2.putText(test, a, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
print(np.argmax(array_l))
'''
'''
data_path ='test_data'
imagePaths = sorted(list(data_prosess.utils_paths.list_images(data_path)))
random.seed(42)
random.shuffle(imagePaths)
for imagePath in imagePaths:
        # 读取图像数据
        test = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        #image = cv2.resize(test, (160, 160))
        image = test / 255
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        label = imagePath.split(os.path.sep)[-2]
        if label == '0':
           pre_label = model.predict(image)
           array_l = np.array(model.predict(image))
           a = "%d"%np.argmax(array_l)
           test = cv2.putText(test, a, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
           image_name = imagePath.split(os.path.sep)[-1]
           path = os.path.join('eval/244_nofine_nodata/0/' , image_name)
           cv2.imwrite(path, test)
        elif label == '1':
           pre_label = model.predict(image)
           array_l = np.array(model.predict(image))
           a = "%d"%np.argmax(array_l)
           test = cv2.putText(test, a, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),3)
           image_name = imagePath.split(os.path.sep)[-1]
           path = os.path.join('eval/244_nofine_nodata/1/', image_name)
           cv2.imwrite(path, test)
        elif label == '2':
               pre_label = model.predict(image)
               array_l = np.array(model.predict(image))
               a = "%d"%np.argmax(array_l)
               test = cv2.putText(test, a, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
               image_name = imagePath.split(os.path.sep)[-1]
               path = os.path.join('eval/244_nofine_nodata/2/', image_name)
               cv2.imwrite(path, test)
        elif label == '3':
           pre_label = model.predict(image)
           array_l = np.array(model.predict(image))
           a = "%d"%np.argmax(array_l)
           test = cv2.putText(test, a, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
           image_name = imagePath.split(os.path.sep)[-1]
           path = os.path.join('eval/244_nofine_nodata/3/', image_name)
           cv2.imwrite(path, test)
        elif label == '4':
           pre_label = model.predict(image)
           array_l = np.array(model.predict(image))
           a = "%d"%np.argmax(array_l)
           test = cv2.putText(test, a, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
           image_name = imagePath.split(os.path.sep)[-1]
           path = os.path.join('eval/244_nofine_nodata/4/', image_name)
           cv2.imwrite(path, test)
        elif label == '5':
           pre_label = model.predict(image)
           array_l = np.array(model.predict(image))
           a = "%d"%np.argmax(array_l)
           test = cv2.putText(test, a, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3)
           image_name = imagePath.split(os.path.sep)[-1]
           path = os.path.join('eval/244_nofine_nodata/5/', image_name)
           cv2.imwrite(path, test)
        elif label == '6':
           pre_label = model.predict(image)
           array_l = np.array(model.predict(image))
           a = "%d"%np.argmax(array_l)
           test = cv2.putText(test, a, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
           image_name = imagePath.split(os.path.sep)[-1]
           path = os.path.join('eval/244_nofine_nodata/6/', image_name)
           cv2.imwrite(path, test)

data_path ='11'
imagePaths = sorted(list(data_prosess.utils_paths.list_images(data_path)))
random.seed(42)
random.shuffle(imagePaths)
for imagePath in imagePaths:
        # 读取图像数据
        test = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        #image = cv2.resize(test, (160, 160))
        image_name = imagePath.split(os.path.sep)[-1]
        path = os.path.join('14/', image_name)
        cv2.imwrite(path, test)
'''