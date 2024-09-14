import pandas as pd

# 读取 .dat 文件中的数据
file_path = 'ratings.dat'

# 读取数据
data = []
with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split('::')
        user_id, item_id, rating, timestamp = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        data.append((user_id, item_id, rating, timestamp))

# 转换数据为 DataFrame
df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])

# 排序
df = df.sort_values(by=["user_id", "timestamp"])

# 生成 user_id 和 item_id 的映射
user_mapping = {user_id: idx for idx, user_id in enumerate(df["user_id"].unique(),start=1)}
item_mapping = {item_id: idx for idx, item_id in enumerate(df["item_id"].unique(),start=1)}

# 应用映射
df["new_user_id"] = df["user_id"].map(user_mapping)
df["new_item_id"] = df["item_id"].map(item_mapping)

# 结果展示
df = df[["new_user_id", "new_item_id", "rating", "timestamp"]]

# 将结果保存到新的 .dat 文件中
output_file_path = 'outputfile_final.dat'
with open(output_file_path, 'w') as file:
    for row in df.itertuples(index=False):
        file.write(f"{row.new_user_id}::{row.new_item_id}::{row.rating}::{row.timestamp}\n")
