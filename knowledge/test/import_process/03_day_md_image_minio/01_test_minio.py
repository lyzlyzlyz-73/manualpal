from minio import Minio

# 初始化客户端
client = Minio(
    endpoint="192.168.6.160:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # 是否使用https
)

if not client.bucket_exists("mybucket"):
    client.make_bucket("mybucket")

# 上传文件
client.fput_object(
    bucket_name="mybucket",
    object_name="images/shark.png",  # Object 名称含路径
    file_path=r"C:\Users\ASUS\Desktop\shark.png",   # 上传的本地文件路径（绝对路径）
    content_type="image/png"     # MIME类型
)

# 上传后访问地址
url = f"http://192.168.6.160:9000/mybucket/images/shark.png"
print(url)
