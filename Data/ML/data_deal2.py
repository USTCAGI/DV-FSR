import pandas as pd

# 读取 .dat 文件中的数据
file_path = 'outputfile_final.dat'

# 读取数据
data = []
with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split('::')
        user_id, item_id, rating, timestamp = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        data.append((user_id, item_id, rating, timestamp))

# 转换数据为 DataFrame
df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])

# 已经按时间戳排序好，故无需再次排序
# 将结果转换为用户-物品序列
user_sequences = df.groupby("user_id")["item_id"].apply(list).reset_index()

# 将最后一个交互放到 test.dat，前面的交互放到 train.dat
train_data = []
test_data = []

for row in user_sequences.itertuples(index=False):
    user_id, item_seq = row.user_id, row.item_id
    train_data.append((user_id, item_seq[:-1]))
    test_data.append((user_id, item_seq[-1]))

# 写入 train.dat 文件
with open('train_after_final.dat', 'w') as train_file:
    for user_id, item_seq in train_data:
        item_seq_str = ' '.join(map(str, item_seq))
        train_file.write(f"{user_id} {item_seq_str}\n")

# 写入 test.dat 文件
with open('test_after_final.dat', 'w') as test_file:
    for user_id, last_item in test_data:
        test_file.write(f"{user_id} {last_item}\n")
