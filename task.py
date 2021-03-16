import tensorflow as tf
from tensorflow.keras.models import load_model
import jsonpickle
import data_utils, email_notifications
import sys
import os
from google.cloud import storage
import datetime
import numpy as np
import jsonpickle
import cv2
from flask import flash,Flask,Response,request,jsonify
import threading
import requests
import time

# IMPORTANT
# ------------------------------------------------------------------------------------------------------------------------------------
# If you're running this container locally and you want to access the API via local browser, use http://172.17.0.2:5000/
# ------------------------------------------------------------------------------------------------------------------------------------

# Starting flask app
# ------------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
# ------------------------------------------------------------------------------------------------------------------------------------

# general variables declaration
# ------------------------------------------------------------------------------------------------------------------------------------
model_name = 'best_model.hdf5'
bucket_name = 'automatictrainingcicd-aiplatform'
class_names = ['Normal','Viral Pneumonia','COVID-19']
global model
# ------------------------------------------------------------------------------------------------------------------------------------

@app.before_first_request
def before_first_request():
    def initialize_job():
        app.logger.info("Starting initial job")
        app.logger.info("Num GPUs Available: "+str(len(tf.config.experimental.list_physical_devices('GPU'))))
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            tf.config.set_soft_device_placement(True)
            tf.debugging.set_log_device_placement(True)
        global model
        # Checking if there's any model saved at testing on GCS
        model_gcs = data_utils.previous_model(bucket_name,model_name)
        # If any model exists at testing, load it, test it on data and use it on the API
        if model_gcs[0] == True:
            model_gcs = data_utils.load_model(bucket_name,model_name)
            if model_gcs[0] == True:
                try:
                    app.logger.info('Loading model file to application.')
                    model = load_model(model_name)
                except Exception as e:
                    app.logger.info('Something went wrong when trying to production model. Exception: '+str(e)+'. Aborting execution.')
                    email_notifications.exception('Something went wrong trying to production model. Exception: '+str(e))
                    sys.exit(1) 
            else:
                email_notifications.exception('Something went wrong when trying to load production model. Exception: '+str(model_gcs[1]))
                app.logger.info('Something went wrong when trying to load production model. Exception: '+str(model_gcs[1])+'. Aborting execution.')
                sys.exit(1)
        if model_gcs[0] == False:
            app.logger.info('There are no artifacts at model registry. Emailing owner and aborting API execution.')
            email_notifications.send_update('There are no artifacts at model registry. Check GCP for more information.')
            sys.exit(1) 
        if model_gcs[0] == None:
            app.logger.info('Something went wrong when trying to check if production model exists. Exception: '+str(model_gcs[1])+'. Aborting execution.')
            email_notifications.exception('Something went wrong when trying to check if production model exists. Exception: '+model_gcs[1]+'. Aborting execution.')
            sys.exit(1)
        app.logger.info('API initialization has ended successfully.')
    thread = threading.Thread(target=initialize_job)
    thread.start()


@app.route('/init', methods=['GET','POST'])
def init():
    print('API initialized.')
    message = {'message': 'API initialized.'}
    response = jsonpickle.encode(response)
    return Response(response=response, status=200, mimetype="application/json")


@app.route('/', methods=['POST'])
def index():
    if request.method=='POST':
        try:
            #Converting string that contains image to uint8
            app.logger.info('Imaged received. Proceeding to process it.')
            image = np.fromstring(request.data,np.uint8)
            image = image.reshape((128,128,3))
            image = [image]
            image = np.array(image)
            image = image.astype(np.float16)
            app.logger.info('Passing image to model.')
            result = model.predict(image)
            result = np.argmax(result)
            app.logger.info('Encoding response')
            message = {'message': '{}'.format(str(result))}
            json_response = jsonify(message)
            app.logger.info(json_response)
            return json_response

        except Exception as e:
            app.logger.info('Something went wrong when trying to make prediction via Production API. Exception: '+str(e)+'. Aborting execution.')
            message = {'message': 'Error'}
            json_response = jsonify(message)
            app.logger.info(json_response)
            email_notifications.exception('Something went wrong when trying to make prediction via Production API. Exception: '+str(e)+'. Aborting execution.')
            return json_response
    else:
        message = {'message': 'Error. Please use this API in a proper manner.'}
        json_response = jsonify(message)
        app.logger.info(json_response)
        return json_response

def self_initialize():
    def initialization():
        global started
        started = False
        while started == False:
            print('Initializing background jobs')
            try:
                server_response = requests.get('http://127.0.0.1:5000/init')
                if server_response.status_code == 200:
                    print('API has started successfully, quitting initialization job.')
                    started = True
            except:
                print('API has not started. Still attempting to initialize it.')
            time.sleep(3)

    print('Initializing API.')
    thread = threading.Thread(target=initialization)
    thread.start()

if __name__ == '__main__':
    self_initialize()
    app.run(host='0.0.0.0',debug=True,threaded=True)