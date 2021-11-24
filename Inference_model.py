import numpy as np
import pandas as pd
import os
import glob
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import cv2 as cv
import os
from sklearn.utils import Bunch
from sklearn.metrics import confusion_matrix
import mmmil.utils.postprocessing as utils

# Variables



Data_dir = './data/Test_40/'
label_csv_path ='./data/label_40.csv'


def model_prediction(x_test, model):
    result = []

    prediction = model.predict(x_test)
    # prediction2 = model2.predict(x_test)

    prediction_pseudo_distance_r = prediction[0][:int(prediction[0].shape[0] / 2), 0]  # Right
    prediction_pseudo_distance_l = prediction[0][int(prediction[0].shape[0] / 2):, 0]  # left

    prediction_class_r = prediction[1][:int(prediction[1].shape[0] / 2)]  # Right
    prediction_class_l = prediction[1][int(prediction[1].shape[0] / 2):]  # left

    prediction_measured_distance_r = prediction[2][:int(prediction[2].shape[0] / 2)]  # Right
    prediction_measured_distance_l = prediction[2][int(prediction[2].shape[0] / 2):]  # left


    result.append(prediction_pseudo_distance_r)
    result.append(prediction_pseudo_distance_l)
    result.append(prediction_class_r)
    result.append(prediction_class_l)
    result.append(prediction_measured_distance_r)
    result.append(prediction_measured_distance_l)

    return result


data_path_list = []
for i in range(len(os.listdir(f'{Data_dir}'))):
    data_path_list.append(glob.glob(os.path.join(f'{Data_dir}',
                                                 os.listdir(f'{Data_dir}')[i],
                                                 '*.jpg'))[0])
patient_list=os.listdir(f'{Data_dir}')

Pr_R = []
Pr_L = []

for i in range(len(data_path_list)):
    malocclusion_predict = utils.malocclusion_result(data_path_list[i])
    Pr_R.append(malocclusion_predict["Right_class"])
    Pr_L.append(malocclusion_predict["Left_class"])
    
    
prediction_result = pd.DataFrame({"patient_id": patient_list,
                                  "Prediction_angle_clss_r": Pr_R,
                                  "Prediction_angle_clss_l": Pr_L },)
# prediction_result.to_csv('prediction_result.csv',index=False)
prediction_y = Pr_R+Pr_L

label_csv = pd.read_csv(label_csv_path)
y_class = label_csv.loc[:, ['Angle_Class_Type_Right', 'Angle_Class_Type_Left']].to_numpy()
label_R = []
label_L = []

for i in range(len(y_class)):
    label_R.append(y_class[i][0])
    label_L.append(y_class[i][1])
label = label_R +label_L


print('class 1 : ',len(np.where(y_class==1)[0]))
print('class 2 : ',len(np.where(y_class==2)[0]))
print('class 3 : ',len(np.where(y_class==3)[0]))
cm = confusion_matrix(label, prediction_y, labels=[1, 2, 3])
print("############################")

print("Confusion matrix")
print(cm)

print("############################")
accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / cm.sum()

print(f'Accuracy: {accuracy:.3f}')
print("############################")

recall_class1 = cm[0, 0] / cm[0].sum()
recall_class2 = cm[1, 1] / cm[1].sum()
recall_class3 = cm[2, 2] / cm[2].sum()
recall = (recall_class1 + recall_class2 + recall_class3) / 3

print(f'Class 1: {recall_class1:.3f}')
print(f'Class 2: {recall_class2:.3f}')
print(f'Class 3: {recall_class3:.3f}')
print(f'Recall (average): {recall:.3f}')

print("############################")


precision_class1 = cm[0, 0] / cm[:, 0].sum()
precision_class2 = cm[1, 1] / cm[:, 1].sum()
precision_class3 = cm[2, 2] / cm[:, 2].sum()
precision = (precision_class1 + precision_class2 + precision_class3) / 3

print(f'Class 1: {precision_class1:.3f}')
print(f'Class 2: {precision_class2:.3f}')
print(f'Class 3: {precision_class3:.3f}')
print(f'Precision (average): {precision:.3f}')
print("############################")
