from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    
    data = pd.read_csv("/Users/aryansingh/jupyter-env/USA_Housing.csv")
    data = data.drop(['Address'], axis = 1)
    features= data.drop(['Price'], axis = 1)
    result = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(features, result, test_size = .30)
    model = LinearRegression()
    model.fit(X_train, y_train)
    n11 = float(request.POST.get('n1') or '0')
    n21 = float(request.POST.get('n2') or '0')
    n31 = float(request.POST.get('n3') or '0')
    n41 = float(request.POST.get('n4') or '0')
    n51 = float(request.POST.get('n5') or '0')
    input_data = np.array([[n11, n21, n31, n41, n51]])
    prediction = model.predict(input_data)
    prediction = round(prediction[0])

    return render(request, 'predict.html', {"result5":str(prediction)})