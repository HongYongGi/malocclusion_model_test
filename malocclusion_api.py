import glob
import os
import pprint
from tqdm import tqdm
from sklearn.utils import Bunch
import numpy as np
import pandas as pd
import requests
import cv2 as cv
from sklearn.metrics import confusion_matrix



# Variables

Data_dir = './data/Test_50/'
label_csv_path ='./data/label_50.csv'



data_path_list = []
for i in range(len(os.listdir(f'{Data_dir}'))):
    data_path_list.append(glob.glob(os.path.join(f'{Data_dir}',
                                                 os.listdir(f'{Data_dir}')[i],'*.jpg'))[0])
data_path_list.sort()
patient_list=os.listdir(f'{Data_dir}')
patient_list.sort()


label_csv = pd.read_csv(label_csv_path)
y_class = label_csv.loc[:, ['Angle_Class_Type_Right', 'Angle_Class_Type_Left']].to_numpy()
label_R = []
label_L = []

for i in range(len(y_class)):
    label_R.append(y_class[i][0])
    label_L.append(y_class[i][1])
label = label_R +label_L


url = 'http://127.0.0.1:8000/MalocclussionInference/v01/'

Pr_R = []
Pr_L = []

for i in range(len(data_path_list)):
    fstr = data_path_list[i]
    with open(fstr,'rb') as files:
        upload = {'Malocclusion_Image':files}
        res = requests.post(url, files=upload)
        Pr_R.append(res.json()["Prediction_Angle_Class_R"])
        Pr_L.append(res.json()["Prediction_Angle_Class_L"])
    print("#########################")
    print(patient_list[i])
    print(f"Label Right : {label_R[i]}, Prediction Right : {res.json()['Prediction_Angle_Class_R']}")
    print(f"Label Left  : {label_L[i]}, Prediction Left  : {res.json()['Prediction_Angle_Class_L']}")
    print("#########################")

prediction_result = pd.DataFrame({"patient_id": patient_list,
                                  "Prediction_angle_clss_r": Pr_R,
                                  "Prediction_angle_clss_l": Pr_L },)

# prediction_result.to_csv('prediction_result.csv',index=False)
prediction_y = Pr_R+Pr_L



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
