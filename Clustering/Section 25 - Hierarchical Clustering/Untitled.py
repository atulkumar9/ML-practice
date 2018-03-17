# coding: utf-8

# # Hierarchical Clustering 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


dataset = pd.read_csv('Mall_Customers.csv')


# In[11]:


dataset


# In[17]:


X = dataset.iloc[:, 3:5].values


# In[19]:


X


# In[22]:


#Using the dendogram to find the optical number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #ward minimize the variance
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()