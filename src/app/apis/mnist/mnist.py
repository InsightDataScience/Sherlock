'''
Created on Jun 6, 2018

Flask API to run minst

@author: runshengsong
'''
# TO DO
# move to helper
import uuid
import time

from flask import Blueprint, request
from flask import jsonify

blueprint = Blueprint('mnist', __name__)

@blueprint.route('/mnist', method=['POST'])
def run_mnist():
    # TO DO
    # load image and run MNIST model here
    header = {
                "module": "MNIST",
                "service": "MNIST",
                "transId":  + '_' + str(int(round(time.time()))),
                "created": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "organization": " ",
                "requestDate": time.strftime("%m/%d/%Y"),
                "transactionId": str(uuid.uuid1()),
                "vmInstance": "" 
            }
    
    return jsonify({
        "header": header,
        "data": "MNIST prediction URL"
        })