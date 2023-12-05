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

def model_create(val1):
  with open("categories_"+str(val1)+".txt", "r") as f:
    Categories= f.read()
    f.close()
  Categories = Categories.split(sep=",")

  flat_data_arr=[]
  target_arr=[]

  datadir='E:/dataset/yellowplate_normal/'+str(val1)
  for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
      img_array=imread(os.path.join(path,img))
      img_resized=resize(img_array,(150,150,3))
      flat_data_arr.append(img_resized.flatten())
      target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
  flat_data=np.array(flat_data_arr)
  target=np.array(target_arr)
  df=pd.DataFrame(flat_data)
  df['Target']=target
  df

  x=df.iloc[:,:-1]
  y=df.iloc[:,-1]
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=500,random_state=77,stratify=y)
  print('Splitted Successfully')

  param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
  svc=svm.SVC(probability=True)
  print("The training of the model is started, please wait for while as it may take few minutes to complete")
  model=GridSearchCV(svc,param_grid)
  model.fit(x_train,y_train)
  print('The Model is trained well with the given images')
  model.best_params_

  pickle.dump(model,open('WE_HATE_ANPR_'+str(val1)+'.p','wb'))
  print("Pickle is dumped successfully")
  pickle.dump(flat_data_arr, open("flat_data_arr_"+str(val1)+".p", "wb"))
  
  with open("target_arr_"+str(val1)+".txt", "w") as f:
    f.write(str(target_arr))
    f.close

  with open("svc_"+str(val1)+".txt", "w") as f:
    f.write(str(svc))
    f.close


  with open("param_grid_"+str(val1)+".txt", "w") as f:
    f.write(str(param_grid))
    f.close

z=threading.Thread(target=model_create,args=("a"))
y=threading.Thread(target=model_create,args=("b"))
## x=threading.Thread(target=model_create,args=("c"))
## w=threading.Thread(target=model_create,args=("d"))
## v=threading.Thread(target=model_create,args=("e"))
## u=threading.Thread(target=model_create,args=("f"))

z.start()
y.start()
## x.start()
## w.start()
## v.start()
## u.start()

z.join()
y.join()
## x.join()
## w.join()
## v.join()
## u.join()
