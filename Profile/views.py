from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

MODEL_PATH = 'C:/FakeProfile/Profile/model/fake_profile_model.h5'

def index(request):
    return render(request, 'index.html', {})

def User(request):
    return render(request, 'User.html', {})

def Admin(request):
    return render(request, 'Admin.html', {})

def AdminLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == 'admin' and password == 'admin':
            context = {'data': 'welcome ' + username}
            return render(request, 'AdminScreen.html', context)
        else:
            context = {'data': 'login failed'}
            return render(request, 'Admin.html', context)

def importdata(): 
    balance_data = pd.read_csv('C:/FakeProfile/Profile/dataset/dataset.txt')
    balance_data = balance_data.abs()
    return balance_data 

def splitdataset(balance_data):
    X = balance_data.values[:, 0:8] 
    y_ = balance_data.values[:, 8]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y_)
    return train_test_split(X, Y, test_size=0.2)

def UserCheck(request):
    if request.method == 'POST':
        data = request.POST.get('t1', False)
        input_data = 'Account_Age,Gender,User_Age,Link_Desc,Status_Count,Friend_Count,Location,Location_IP\n' + data + "\n"
        
        with open("C:/FakeProfile/Profile/dataset/test.txt", "w") as f:
            f.write(input_data)
        
        test = pd.read_csv('C:/FakeProfile/Profile/dataset/test.txt')
        test = test.values[:, 0:8] 

        if not os.path.exists(MODEL_PATH):
            return render(request, 'User.html', {'data': 'Model not trained. Please generate the model first.'})

        model = load_model(MODEL_PATH)
        predict = model.predict(test)
        predict_class = predict.argmax(axis=-1)
        
        msg = "Given Account Details Predicted As Genuine" if predict_class[0] == 0 else "Given Account Details Predicted As Fake"
        return render(request, 'User.html', {'data': msg})

def GenerateModel(request):
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)
    
    model = Sequential([
        Dense(200, input_shape=(8,), activation='relu', name='fc1'),
        Dense(200, activation='relu', name='fc2'),
        Dense(2, activation='softmax', name='output')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('CNN Neural Network Model Summary: ')
    print(model.summary())
    
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    results = model.evaluate(test_x, test_y)
    ann_acc = results[1] * 100
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    model.save(MODEL_PATH)
    
    context = {'data': f'ANN Accuracy : {ann_acc:.2f}%'}
    return render(request, 'AdminScreen.html', context)

def ViewTrain(request):
    data = pd.read_csv('C:/FakeProfile/Profile/dataset/dataset.txt')
    rows = data.shape[0]
    cols = data.shape[1]
    
    headers = ['Account Age', 'Gender', 'User Age', 'Link Description', 'Status Count', 'Friend Count', 'Location', 'Location IP', 'Profile Status']
    
    table = '<table border=1 align=center width=100%><tr>'
    table += ''.join([f'<th><font size=4 color=white>{header}</th>' for header in headers])
    table += '</tr>'
    
    for i in range(rows):
        table += '<tr>'
        for j in range(cols):
            table += f'<td><font size=3 color=white>{data.iloc[i,j]}</font></td>'
        table += '</tr>'
    
    table += '</table>'
    
    context = {'data': table}
    return render(request, 'ViewData.html', context)