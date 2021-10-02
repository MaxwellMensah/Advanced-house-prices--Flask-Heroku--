#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle  #Initialize the flask App


app = Flask(__name__)

model = pickle.load(open('model1.pkl', 'rb'))

#default page of our web-
# background-image: url({{ url_for('static', filename='img/home.jpg') }})
@app.route('/')
def home():
    return render_template('index.html')  #"<h1>Hello</h1>"


#To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    #To use the predict button in our web-app
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='SalePrice of the house is: {}'.format(output)) 
# mOve to next page / render the predciton on a new page.

    

if __name__ == "__main__":
    app.run(debug = True) 
    
# issues: flask not debugger off
# model.pkl not saving model in flask
# sklearn install in flask directory for py
# reinstall flask in deployment dir if possible :: cd C:\Users\Maxie\Deployment\flask
# design UI (houses background)
# dockerize it
# heroku free cloud deployment
# create requirement txt..pip freeze > requirement.txt
# FUTURE: github/jenkins cicd/aws ebr stalk/ec2 extension