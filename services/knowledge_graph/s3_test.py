from boto3 import client, Session

sess = Session(profile_name="staging2")
bucket_name = "io.etherlabs.staging2.contexts"
conn = sess.client("s3")

dir_name = "6baa3490/context-instance-graphs/"

folder_list = "/"
paginator = conn.get_paginator("list_objects_v2")
s3_results = paginator.paginate(
    Bucket=bucket_name, Prefix="01", Delimiter="/", PaginationConfig={"PageSize": 1000}
)

print(s3_results)
bucket_object_list = []
for page in s3_results:
    if "CommonPrefixes" in page:
        for key in page["CommonPrefixes"]:
            s3_file_name = key["Prefix"].split("/")[0]
            bucket_object_list.append(s3_file_name)
    break
print(bucket_object_list)


# Connect to a specific bucket
bucket = conn.get_bucket(bucket_name)

# Get subdirectory info
for key in bucket.list(prefix="sub_directory/", delimiter="/"):
    print(key.name)
