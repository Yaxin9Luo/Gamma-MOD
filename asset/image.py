from PIL import Image

def process_image(input_path, output_path, new_size=(300, 300)):
    # 打开图片
    img = Image.open(input_path)
    # 调整图片大小
    img_resized = img.resize(new_size, Image.LANCZOS)
    
    # 保存处理后的图片
    img_resized.save(output_path)

# 使用函数
input_image = "gamma_logo.jpg"
output_image = "gamma_logo_processed.jpg"
process_image(input_image, output_image)