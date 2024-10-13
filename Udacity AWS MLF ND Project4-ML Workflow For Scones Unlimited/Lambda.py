# Lambda Function 1

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    s3.download_file(bucket, key, "/tmp/image.png")
    
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }




# Lambda Function 2

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor

ENDPOINT = "image-classification-2024-09-12-03-30-11-543"

def lambda_handler(event, context):
    image = base64.b64decode(event['image_data'])

    predictor = Predictor(endpoint_name=ENDPOINT)

    predictor.serializer = IdentitySerializer("image/png")

    inferences = predictor.predict(image)

    event["inferences"] = json.loads(inferences.decode('utf-8'))
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }





# Lambda Function 3

import json

THRESHOLD = 0.70


def lambda_handler(event, context):
    
    inferences = event['inferences']
    
    meets_threshold = max(list(inferences))>THRESHOLD
    
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }