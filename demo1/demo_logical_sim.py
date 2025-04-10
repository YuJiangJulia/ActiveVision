import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = "your_image.jpg"
task_prompt = "Find the object used for writing"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
cv2.circle(heatmap, (center_x, center_y), 60, 255, -1)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

plt.figure(figsize=(8, 6))
plt.imshow(overlay)
plt.title(f"Prompt: {task_prompt}", fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.show()
