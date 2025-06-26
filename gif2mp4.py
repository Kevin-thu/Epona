# 导入moviepy.editor模块
import os
import moviepy.editor as mp

# 定义一个函数，接受一个文件夹路径作为参数，将该文件夹下的所有gif转换为mp4
def convert_gif_to_mp4(folder_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    # 遍历文件夹下的所有文件
    for file in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, file)
        # 判断文件是否是gif格式
        if file_path.endswith(".gif"):
            # 使用moviepy的VideoFileClip对象，创建一个视频剪辑
            clip = mp.VideoFileClip(file_path)
            # 将视频剪辑写入一个mp4文件，文件名与gif文件相同，只是后缀改为mp4
            clip.write_videofile(os.path.join(target_path, file.replace(".gif", ".mp4")))
            # 关闭视频剪辑对象，释放资源
            clip.close()

# 调用函数，传入你想要转换的文件夹路径，例如"C:\\Users\\gif_folder"
convert_gif_to_mp4(".", "..\\videos")
