#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing the libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
#For reading the pickle file
import pickle


# In[4]:


# Create a flask app
app = Flask(__name__,template_folder='Template')
#Load the pickle file
model = pickle.load(open('model.pkl','rb'))


# In[5]:


# For Rendering to web home page
@app.route('/')
def home():
    return render_template('index.html')


# In[6]:


# Providing input for getting output using POST
@app.route('/predict',methods=['POST'])
def predict():
    '''
    Rendering results on HTML GUI
    '''
    #take input from all the forms
    in_features = [int(x) for x in request.form.values()]
    final_features = [np.array(in_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text='Salary of Employee should be $ {}'.format(output))


# In[7]:


#for running the flask
if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




