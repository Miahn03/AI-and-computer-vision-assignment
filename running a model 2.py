import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle
import threading
def model_run(val1):
    model=pickle.load(open('WE_HATE_ANPR_'+str(val1)+'.p','rb'))
    flat_data_arr = pickle.load(open("flat_data_arr_"+str(val1)+".p", "rb"))
    with open("categories_"+str(val1)+".txt", "r") as f:
        Categories= f.read()
        f.close()
    with open("target_arr_"+str(val1)+".txt", "r") as f:
        target_arr= f.read()
        f.close()
    with open("svc_"+str(val1)+".txt", "r") as f:
        svc= f.read()
        f.close()
    with open("param_grid_"+str(val1)+".txt", "r") as f:
        param_grid= f.read()
        f.close()
    Categories = Categories.split(sep=",")
    url=input('Enter URL of Image')
    img=imread(url)
    plt.imshow(img)
    plt.show()
    img_resize=resize(img,(150,150,3))
    l=[img_resize.flatten()]
    probability=model.predict_proba(l)
    for ind,val in enumerate(Categories):
        print(f'{val} = {probability[0][ind]*100}%')
    print("The predicted image is : "+Categories[model.predict(l)[0]])
    print(f'Is the image a {Categories[model.predict(l)[0]]} ?(y/n)')
    while(True):
        b=input()
        if(b=="y" or b=="n"):
            break
        print("please enter either y or n")

    if(b=='n'):
        print("What is the image?")
        for i in range(len(Categories)):
            print(f"Enter {i} for {Categories[i]}")
        k=int(input())
        while(k<0 or k>=len(Categories)):
            print(f"Please enter a valid number between 0-{len(Categories)-1}")
            k=int(input())
        print("Please wait for a while for the model to learn from this image :)")
        flat_arr=flat_data_arr.copy()
        tar_arr=target_arr.copy()
        tar_arr.append(k)
        flat_arr.extend(l)
        tar_arr=np.array(tar_arr)
        flat_df=np.array(flat_arr)
        df1=pd.DataFrame(flat_df)
        df1['Target']=tar_arr
        model1=GridSearchCV(svc,param_grid)
        x1=df1.iloc[:,:-1]
        y1=df1.iloc[:,-1]
        x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.20,random_state=77,stratify=y1)
        d={}
        for i in model.best_params_:
            d[i]=[model.best_params_[i]]
        model1=GridSearchCV(svc,d)
        model1.fit(x_train1,y_train1)
        y_pred1=model.predict(x_test1)
        print(f"The model is now {accuracy_score(y_pred1,y_test1)*100}% accurate")
        pickle.dump(model1,open('img_model.p','wb'))
    print("Thank you for your feedback")
model_run(val1="a")