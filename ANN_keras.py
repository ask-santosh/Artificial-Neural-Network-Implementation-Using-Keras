#!/usr/bin/env python
# coding: utf-8

# In[15]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[16]:


import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler


# In[17]:


train_sample = []
train_label = []


# In[18]:


for i in range(1000):
    younger_ages = randint(13,64)
    train_sample.append(younger_ages)
    train_label.append(0)
    
    older_ages = randint(65,100)
    train_sample.append(older_ages)
    train_label.append(1)
#     print(train_label)


# In[19]:


train_sample = np.array(train_sample)
print(type(train_sample))


# In[20]:


train_label = np.array(train_label)
print(train_label)


# In[21]:


scaler = MinMaxScaler(feature_range=(0,1))
scaler_train_sample = scaler.fit_transform(train_sample.reshape(-1,1))
scaler_train_sample


# In[22]:


model = Sequential([Dense(16, input_dim=1, activation = 'relu'), Dense(32, activation='relu'), Dense(2,activation = 'softmax')])
model.summary()


# In[23]:


model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[24]:


history = model.fit(train_sample, train_label, batch_size=10, epochs = 15, validation_split= 0.2)
print(history.history.keys())
import matplotlib.pyplot as plt
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,11)
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[25]:


test_sample = []
test_label = []
for i in range(500):
    younger_ages = randint(13,64)
    test_sample.append(younger_ages)
    test_label.append(0)
    
    older_ages = randint(65,100)
    test_sample.append(older_ages)
    test_label.append(1)
    print(test_label)


# In[26]:


test_sample = np.array(test_sample)
test_label = np.array(test_label)
test_sample[38]


# In[27]:


output_model=model.predict_classes(test_sample, batch_size=10)


# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


predicted_values =confusion_matrix(test_label, output_model) 


# In[30]:


predicted_values


# In[ ]:




