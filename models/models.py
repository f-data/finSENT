#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional, Dropout, Flatten, GRU
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

import matplotlib.pyplot as plt

import numpy as np


# In[ ]:


weight_decay = 1e-4
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[ ]:


class BaseDnnClassifier:
    def fit(self,X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, nb_epoch = epochs, batch_size=batch_size)
    
    def predict_proba(self,X_test):
        return self.model.predict(X_test)
    
    def predict(self,X_test):
        return np.argmax(self.model.predict(X_test),axis=1)
    
    def evaluate(self,X_test,y_test):
        y_pred=self.model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel() 
        mcc = matthews_corrcoef(Y_dev_labels, Y_pred)
        return tn,fp,fn,tp,mcc;
        


# In[ ]:


class MLPClassifier(BaseDnnClassifier):
    def __init__(self,units=0,vector_size=0,embedding_matrix=None,max_features=0, embed_size=0):
        self.model = Sequential()
        if (embedding_matrix!=null):
            self.model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False))
            self.model.add(Dense(units, activation='relu'))
        else:
            self.model.add(Dense(units, activation='relu', input_shape=(vector_size,)))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# In[109]:


class LSTMClassifier(BaseDnnClassifier):
    def __init__(self,units,input_shape=(64,1,),embedding_matrix=None,max_features=0, embed_size=0):
        input_ = Input(shape=input_shape)
        if (embedding_matrix!=null):
            input_ = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(input_)
        x1 = LSTM(units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)(input_) 
        x1 = Flatten()(x1)
        out = Dense(2, activation='softmax')(x1)
        self.model = Model(inputs=[input_], outputs=out)
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# In[110]:


class BiLSTMClassifier(BaseDnnClassifier):
    def __init__(self,units,input_shape=(64,1,),embedding_matrix=None,max_features=0, embed_size=0):
        input_ = Input(shape=input_shape)
        if (embedding_matrix!=null):
            input_ = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(input_)
        x1 = Bidirectional(LSTM(units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(input_)
        x1 = Flatten()(x1)
        out = Dense(2, activation='softmax')(x1)
        self.model = Model(inputs=[input_], outputs=out)
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# In[111]:


class BiGRUClassifier(BaseDnnClassifier):
    def __init__(self,units,input_shape=(64,1,),embedding_matrix=None,max_features=0, embed_size=0):
        input_ = Input(shape=input_shape)
        if (embedding_matrix!=null):
            input_ = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(input_)
        x1 = Bidirectional(GRU(units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(input_)
        x1 = Flatten()(x1)
        out = Dense(2, activation='softmax')(x1)
        self.model = Model(inputs=[input_], outputs=out)
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# In[112]:


class BiLSTMAttClassifier(BaseDnnClassifier):
    def __init__(self,units,input_shape=(64,1,),maxlen=64,embedding_matrix=None,max_features=0, embed_size=0):
        input_ = Input(shape=input_shape)
        if (embedding_matrix!=null):
            input_ = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(input_)
        x1 = Bidirectional(LSTM(units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(input_)
        x1 = Attention(maxlen)(x1)
        out = Dense(2, activation='softmax')(x1)
        self.model = Model(inputs=[input_], outputs=out)
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# In[125]:


class BiGRUAttClassifier(BaseDnnClassifier):
    def __init__(self,units,input_shape=(64,1,),useEmb=False,maxlen=64,embedding_matrix=None,max_features=0, embed_size=0):
        input_ = Input(shape=input_shape)
        if (useEmb==True):
            input_ = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(input_)
        x1 = Bidirectional(GRU(units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(input_)
        x1 = Attention(maxlen)(x1)
        out = Dense(2, activation='softmax')(x1)
        self.model = Model(inputs=[input_], outputs=out)
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

