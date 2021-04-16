hidden_count=100
epochs=200
batch_size=1024
activation='relu'

#Libraries
import time
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as backend

#Generate 1000 samples
X = np.random.rand(1000,1)
y = X**5 - 2*X**3 + 6*X**2 + 10*X - 5

#Shuffle and split data into train set and test set
data = train_test_split(X,y)
    
#Record the start time
start = time.time()

#Unpack the data
X_train, X_test, y_train, y_test = data

#Layers
inputs = Input(shape=(X_train.shape[1],))
x = Dense(hidden_count, activation=activation)(inputs)
predictions = Dense(1, activation='linear')(x)

#Model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mean_squared_error')
model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=0) #Do not display progress

#Collect and display info
param_count = model.count_params()
loss_tr = round(model.evaluate(x=X_train,y=y_train,batch_size=batch_size,verbose=0),4)
loss_te = round(model.evaluate(x=X_test,y=y_test,batch_size=batch_size,verbose=0),4)
elapsed = round(time.time() - start,2)    
print("Hidden count:",str(hidden_count).ljust(5),
      "Parameters:",str(param_count).ljust(6),
      "loss (train,test):",str(loss_tr).ljust(7),str(loss_te).ljust(7),
      "Time:",str(elapsed)+"s",
     )

backend.clear_session()