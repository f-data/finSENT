#!/usr/bin/env python
# coding: utf-8

# In[34]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
import itertools 
import numpy as np


# In[35]:


class Evaluator:
    def __init__(self,y_true,y_pred):
        self.y_true = y_true;
        self.y_pred = y_pred;
    
    def _eval(self):
        return confusion_matrix(self.y_true,self.y_pred).ravel() 
    
    def acc(self):
        return accuracy_score(self.y_true,self.y_pred)
    
    def precision(self):
        return precision_score(self.y_true,self.y_pred)
    
    def recall(self):
        return recall_score(self.y_true,self.y_pred)
    
    def specificity(self):
        tn, fp, fn, tp = self._eval()
        return tn/(tn+fp);
    
    def f1(self):
        return f1_score(self.y_true,self.y_pred)
    
    def mcc(self):
        return matthews_corrcoef(self.y_true,self.y_pred)
    
    def report(self):
        return self.acc(),self.precision(),self.recall(),self.specificity(),self.f1(),self.mcc();
    
    def print_report(self):
        a,p,r,s,f,m=self.report();
        print("Accuracy: %.4f" % a);
        print("Precision: %.4f" % p);
        print("Recall: %.4f" % r);
        print("Specificity: %.4f" % s);
        print("F1-Score: %.4f" % f);
        print("MCC: %.4f" % m);
        
    def plot_confusion_matrix(self, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        cm = confusion_matrix(self.y_true,self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.tight_layout()
        

