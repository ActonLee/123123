import os
import shutil

def copy_files(src_folder, dest_folder, file_name):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的子文件夹
    for root, dirs, files in os.walk(src_folder):  
        if dirs != []:
            for dir in dirs:
                os.makedirs(os.path.join(dest_folder, dir), exist_ok=True)     
        for file in files:
            # 检查文件名是否匹配
            if file == file_name:
                
                # 构建源文件的完整路径
                src_file_path = os.path.join(root, file)
                
                # 构建目标文件的完整路径
                dest_file_path = os.path.join(dest_folder, root[-12:], file)
                
                # 复制文件
                shutil.copy(src_file_path, dest_file_path)
                print(f'复制文件: {src_file_path} 到 {dest_file_path}')

# 设置主文件夹路径和目标文件夹路径
main_folder = './garment_dataset/shirt_dataset_rest'
destination_folder = './garment_dataset_clean'
file_name = 'shirt_mesh_r_clean.obj'

# 执行复制文件操作
copy_files(main_folder, destination_folder, file_name)

