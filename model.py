# Import Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
import matplotlib.pyplot as plt



# Preprocessing Data
data = pd.read_excel('D:\Final Year Project\Stocks Data\RELIANCE\Reliance train.xlsx')
data["Close"]= pd.to_numeric(data.Close, errors='coerce')
data = data.dropna()
trainData = data.iloc[:,4:5].values
sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)


#Training The Data
X_train = []
y_train = []
for i in range (50,496):
    X_train.append(trainData[i-50:i,0])
    y_train.append(trainData[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)
X_forecast=np.array(X_train[-1,1:])
X_forecast=np.append(X_forecast,y_train[-1])
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
#LSTM Model
model = Sequential()

model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
model.add(Dropout(0.5))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile(optimizer='adam',loss="mean_squared_error")
hist = model.fit(X_train, y_train, epochs = 20, batch_size = 32, verbose=2)
# Forecasting Prediction
forecasted_stock_price = model.predict(X_forecast)

# Getting original prices back from scaled values
forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
print(forecasted_stock_price)
plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

testData = pd.read_excel('D:\Final Year Project\Stocks Data\RELIANCE\Reliance train.xlsx')
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[30:,0:].values

inputClosing = testData.iloc[:,0:].values
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape
X_test = []
length = len(testData)
timestep = 30
for i in range(timestep,length):
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
y_pred = model.predict(X_test)
predicted_price = sc.inverse_transform(y_pred)
plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Reliance stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

model.save('Reliance.keras')



