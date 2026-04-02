import os
from PIL import Image

def main():
    # 定義輸入圖片的檔案名
    image_files = [f"r_{i}.png" for i in range(6)]
    
    # 確認檔案是否都存在
    for file in image_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"檔案 {file} 不存在")

    # 打開所有圖片
    images = [Image.open(file) for file in image_files]

    # 確保所有圖片大小一致
    widths, heights = zip(*(img.size for img in images))
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        raise ValueError("所有圖片必須具有相同的大小")

    img_width, img_height = images[0].size

    # 設定大圖的大小 (3x2 結構)
    grid_width = 3
    grid_height = 2
    total_width = grid_width * img_width
    total_height = grid_height * img_height

    # 建立空白的大圖
    result = Image.new("RGB", (total_width, total_height))

    # 將每張小圖貼上去
    for idx, img in enumerate(images):
        x_offset = (idx % grid_width) * img_width
        y_offset = (idx // grid_width) * img_height
        result.paste(img, (x_offset, y_offset))

    # 儲存結果
    result.save("output.png")
    print("已成功將圖片合併並儲存為 output.png")

if __name__ == "__main__":
    main()
