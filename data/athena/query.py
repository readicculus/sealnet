import csv
import os
import pandas as pd

import boto3
import botocore
from retrying import retry
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
database = "sealnet"

s3_bucket = "arcticseals-athena"
s3_output = "s3://" + s3_bucket

athena = boto3.client('athena')
s3 = boto3.resource('s3')

@retry(stop_max_attempt_number = 10,
    wait_exponential_multiplier = 300,
    wait_exponential_max = 1 * 60 * 1000)
def poll_status(_id):
    result = athena.get_query_execution( QueryExecutionId = _id )
    state  = result['QueryExecution']['Status']['State']

    if state == 'SUCCEEDED':
        return result
    elif state == 'FAILED':
        return result
    else:
        raise Exception

def cleanup_query(local_filename, s3_key):
    # delete result file
    if local_filename is not None and os.path.isfile(local_filename):
        os.remove(local_filename)
    if s3_key is not None:
        try:
            obj = s3.Object(s3_bucket, s3_key)
            obj.delete()
        except:
            print("Unable to delete QueryObject")

def run_query(query, database, s3_output):
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': s3_output,
    })

    QueryExecutionId = response['QueryExecutionId']
    result = poll_status(QueryExecutionId)

    if result['QueryExecution']['Status']['State'] == 'SUCCEEDED':
        print("Query SUCCEEDED: {}".format(QueryExecutionId))

        s3_key = QueryExecutionId + '.csv'
        local_filename = QueryExecutionId + '.csv'

        # download result file
        try:
            s3.Bucket(s3_bucket).download_file(s3_key, local_filename)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        res = {
            # "rows": rows,
            "query_id":QueryExecutionId,
            "s3_key": s3_key,
            "local_filename": local_filename,
            "query_response": response
        }
        return res

def run_query_make_df(query, database, s3_output):
    res = run_query(query, database, s3_output)
    df = pd.read_csv(res["local_filename"])
    res["df"] = df
    return res

