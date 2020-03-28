#Importing necessary packages
# import numpy as np
# from flask import Flask, request, render_template
# import pickle
# from fastai.tabular import *
# import os

# #Saving the working directory and model directory
# cwd = os.getcwd()
# path = cwd + '/model'

# #Initializing the FLASK API
# app = Flask(__name__)

# #Loading the saved model using fastai's load_learner method
# model = load_learner(path, 'export.pkl')

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# model_file_url = 'https://drive.google.com/uc?export=download&id=1zBFIX-O5xnFDNspJpy83Uqjg-w48QQ56'
# model_file_name = 'export'
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

# async def download_file(url, dest):
#     if dest.exists(): return
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             data = await response.read()
#             with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    # await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    learn = load_learner(path/'model','export.pkl')
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()
    


#Defining the home page for the web service
# @app.route('/')
# def home():
#     return render_template('myindex.html')

@app.route('/')
def index(request):
    html = path/'view'/'myindex.html'
    return HTMLResponse(html.open().read())

#Writing api for inference using the loaded model
@app.route('/predict',methods=['POST'])

#Defining the predict method get input from the html page and to predict using the trained model

async def predict(request):
    
    try:
    	#all the input labels . We had only trained the model using these selected features.
        
        labels = ['age', 'sex', 'cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']

        #Collecting values from the html form and converting into respective types as expected by the model
        Age =  await int(request.form["age"])
        Sex =  await request.form["sex"]
        Cough = await request.form["cough"]
        Fever =  await request.form["fever"]
        Chills = await request.form["chills"]
        Sore_throat = await request.form["sore_throat"]
        Headache =  await request.form["headache"]
        Fatigue = await request.form["fatigue"]



        features = [Age, Sex, Cough, Fever,Chills, Sore_throat, Headache, Fatigue]

        #fastai predicts from a pandas series. so converting the list to a series
        to_predict = pd.Series(features, index = labels)

        #Getting the prediction from the model and rounding the float into 2 decimal places
        prediction = f' Please wait for {int(round(float(learn.predict(to_predict)[1]),0))} days'

        return JSONResponse({'result': prediction})
    
    except Exception as e:
        print(e)

        
        # Making all predictions below 0 lakhs and above 200 lakhs as invalid
    #     if features[2:] == ['No','No','No','No','No','No']:
    #         return render_template('myindex.html', prediction_text= f"You don't show any discernable symptoms")
    #     elif prediction != 0:
    #         return render_template('myindex.html', prediction_text= f'Please wait for {prediction} days before you see a medical professional')
    #     else:
    #         return render_template('myindex.html', prediction_text='You may need immediate assistance')

    # except Exception as e:
    #     return render_template('myindex.html', prediction_text= f'your input {features} is invalid')

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

