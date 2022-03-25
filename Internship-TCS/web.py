from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('mobile.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])

def predict():
    
    
        BP=float(request.values['BP'])
        CS=float(request.values['CS'])
        IM=float(request.values['IM'])
        MD=float(request.values['MD'])
        MW=float(request.values['MW'])
        PH=float(request.values['PH'])
        PW=float(request.values['PW'])
        RM=float(request.values['RM'])
        SH=float(request.values['SH'])
        SW=float(request.values['SW'])
        TT=float(request.values['TT'])
        input=np.array([BP,CS,IM,MD,MW,PH,PW,RM,SH,SW,TT])
        input=np.reshape(input,(1,input.size))
        output=model.predict(input)
        print(output)
        for x in output:
            if(x==3):
                output='very high cost'
                rank=1
            elif(x==2):
                output='high cost'
                rank=2
            elif(x==1):
                output='medium cost'
                rank=3
            elif(x==0):
                output='low cost'
                rank=4
        return render_template ('result.html',prediction_text="The price range is {} and the rank is {}".format(output,rank)) 
if __name__ == "__main__":
    app.run(port=8000)
    
    
        
              
    
   
    
        
        
        
   
   
    
   
       
   
   


