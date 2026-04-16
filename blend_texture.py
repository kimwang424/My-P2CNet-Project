import cv2
import os
import numpy as np

# --- “写实派”终极设置区 ---
raw_img_path = './demo/my_test_01.png' 
blurry_img_path = './p2c_temp/my_test_01.png' 
sharp_img_path = './final_HD_results/my_test_01_out.png' 

# 💡 这里的 alpha 降到 0.3！我们要 70% 的原始真实纹理！
# 这样虽然会模糊一点，但绝对更自然、更有生物质感！
alpha = 0.3 
# ------------------

def realistic_blending():
    img_raw = cv2.imread(raw_img_path)
    img_blurry = cv2.imread(blurry_img_path)
    img_sharp = cv2.imread(sharp_img_path)

    if img_raw is None or img_sharp is None:
        print("❌ 路径不对，太奶检查一下！")
        return

    orig_h, orig_w = img_raw.shape[:2]
    
    # 纠正比例
    img_sharp_resized = cv2.resize(img_sharp, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    img_blurry_resized = cv2.resize(img_blurry, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    # 温和混合 (主要保留真实感)
    final_img = cv2.addWeighted(img_sharp_resized, alpha, img_blurry_resized, 1 - alpha, 0)

    # 4. [关键] 手动增加对比度和饱和度，让鱼看起来有肉感
    # 我们把对比度调得更高，但亮度压一点
    final_img = cv2.convertScaleAbs(final_img, alpha=1.2, beta=0)

    save_path = './THE_FINAL_REALISTIC.png'
    cv2.imwrite(save_path, final_img)
    print(f"🏆 写实版完成！虽然模糊但自然，请查看：{save_path}")

if __name__ == '__main__':
    realistic_blending()