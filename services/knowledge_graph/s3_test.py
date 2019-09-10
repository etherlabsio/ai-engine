from boto3 import client, Session

sess = Session(profile_name="staging2")
bucket_name = "io.etherlabs.staging2.contexts"
conn = sess.client("s3")

dir_name = "6baa3490/context-instance-graphs/"
paginator = conn.get_paginator("list_objects_v2")
s3_results = paginator.paginate(
    Bucket=bucket_name, Prefix=dir_name, PaginationConfig={"PageSize": 1000}
)

print(s3_results)
bucket_object_list = []
for page in s3_results:
    if "Contents" in page:
        for key in page["Contents"]:
            s3_file_name = key["Key"].split("/")[-1]
            bucket_object_list.append(s3_file_name)

print(bucket_object_list)
