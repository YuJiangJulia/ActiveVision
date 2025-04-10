import cv2
import numpy as np

def dummy_heatmap(image):
    img = np.array(image)
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # 模拟一个中心激活区
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    cv2.circle(heatmap, (center_x, center_y), 50, 255, -1)

    # 应用热图颜色映射
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    plt.imshow(overlay)
    plt.title("Simulated Attention")
    plt.axis("off")
    plt.show()

dummy_heatmap(image)
