from flask import Flask,request,render_template,jsonify
import numpy as numpy
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
#import model and scaler 
model=pickle.load(open('models/log_reg_01.pkl','rb'))
scaler=pickle.load(open('models/scaler_feslc.pkl','rb'))

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
     if request.method=='POST':
        cp = int(request.form.get('cp'))
        thalachh = int(request.form.get('thalachh'))
        chol = int(request.form.get('chol'))
        fbs=int(request.form.get('fbs'))
        caa = int(request.form.get('caa'))

        new_data_scaled=scaler.transform([[cp,thalachh,chol,fbs,caa]])
        result=model.predict(new_data_scaled)


        return render_template('index.html',result=result[0])
     
     else:
        return render_template('index.html')




if __name__=="__main__":
    app.run(host="0.0.0.0")