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
url=str(input("URL: "))

def model_run(val1, url=url):
  global x
  global m
  x = 0

  with open("categories_"+str(val1)+".txt", "r") as f:
    Categories = f.read()
    f.close()

  Categories = Categories.split(sep=",")
  model=pickle.load(open('WE_HATE_ANPR_'+str(val1)+'.p','rb'))


  img=imread(url)
  img_resize=resize(img,(150,150,3))
  l=[img_resize.flatten()]
  probability=model.predict_proba(l)
  for ind,val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100}%')
    if (probability[0][ind]*100) > x:
      x = probability[0][ind]*100
      m = Categories[model.predict(l)[0]]

  
  
  



z=threading.Thread(target=model_run,args=("a"))
y=threading.Thread(target=model_run,args=("b"))

z.start()
y.start()

z.join()
y.join()

print("The predicted license plate is : "+m)
print("Is the license plate "+m+"?(y/n)")
while(True):
  b=input()
  if(b=="y" or b=="n"):
    break
  print("please enter either y or n")
