import os
from PIL import Image

def main():
    # 定義輸入圖片的檔案名
    input_path = "../image/output_bb/output.png"

    # 確認輸入圖片是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"檔案 {input_path} 不存在")

    # 開啟輸入圖片
    image = Image.open(input_path)

    # 設定切割的網格大小 (2x3 結構)
    grid_width = 3
    grid_height = 2

    # 計算每張小圖的大小
    img_width, img_height = image.size
    small_width = img_width // grid_width
    small_height = img_height // grid_height

    # 建立輸出資料夾 (若不存在則創建)
    output_folder = os.path.dirname(input_path)

    # 切割圖片並儲存
    for row in range(grid_height):
        for col in range(grid_width):
            left = col * small_width
            upper = row * small_height
            right = left + small_width
            lower = upper + small_height

            # 裁切圖片
            cropped_image = image.crop((left, upper, right, lower))

            # 儲存小圖片
            output_file = os.path.join(output_folder, f"r_{row * grid_width + col}.png")
            cropped_image.save(output_file)

    print(f"已成功將大圖片切割並儲存到 {output_folder}")

if __name__ == "__main__":
    main()
