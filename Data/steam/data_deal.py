def increment_and_save(input_file, output_file):
    # 打开输入文件和输出文件
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 创建一个新的列表来保存处理后的数据
    incremented_lines = []
    
    for line in lines:
        # 拆分每一行的数字，转换为整型，加 1，再转回字符串
        incremented_numbers = ' '.join(str(int(num) + 1) for num in line.split())
        incremented_lines.append(incremented_numbers)
    
    # 写入输出文件
    with open(output_file, 'w') as file:
        for line in incremented_lines:
            file.write(line + '\n')

# 使用这个函数，传递输入文件和要创建的.dat文件的路径
increment_and_save('test.txt', 'output_test.dat')