import csv
import os

import boto3
import botocore
from boto.file import Key
from retrying import retry

database = "sealnet"

s3_bucket = "arcticseals-athena"
s3_output = "s3://" + s3_bucket

athena = boto3.client('athena')
s3     = boto3.resource('s3')

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

def run_query(query, database, s3_output,
              delete_result_s3_query_obj = False):
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
        if delete_result_s3_query_obj:
            try:
                obj = s3.Object(s3_bucket, s3_key)
                obj.delete()
            except:
                print("Unable to delete QueryObject")

        # read file to array
        rows = []
        with open(local_filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)

        # delete result file
        if os.path.isfile(local_filename):
            os.remove(local_filename)

        return rows

if __name__ == '__main__':
    # SQL Query to execute
    query = ("SELECT * FROM lila_original")

    print("Executing query: {}".format(query))
    result = run_query(query, database, s3_output)

    print("Results:")
    print(result)