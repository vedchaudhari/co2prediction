import numpy as np
from flask import Flask,render_template,request
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

#predict button 
@app.route('/predict',methods=['POST'])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0],2)
        return render_template('index.html',prediction_text = 'CO2 Emission of vehicle is : {}'.format(output))
    except Exception as e:
        return render_template('index.html',error_message='Error : {}'.format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)