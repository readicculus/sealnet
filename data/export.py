import pandas as pd
from data.athena.query import run_query_make_df, database, s3_output, cleanup_query

def join_images(path, local=True):
    return


if __name__ == '__main__':
    # SQL Query to execute
    query = ("SELECT * FROM  updated_seals_with_bears WHERE species_id='Polar Bear'")
    query2 = ("SELECT * FROM  lila_updated_all WHERE species_id='Polar Bear'")

    print("Executing query: {}".format(query))
    res = run_query_make_df(query, database, s3_output)
    res2 = run_query_make_df(query2, database, s3_output)

    print("Results:")
    print(res)
    df = res["df"]
    df2 = res2["df"]

    df_concat = pd.concat([df, df2], ignore_index=True)
    cleanup_query(res["local_filename"], res["s3_key"])
    cleanup_query(res2["local_filename"], res2["s3_key"])