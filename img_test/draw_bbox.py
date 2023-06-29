import cv2


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    x, y, w, h = bbox
    # (x1, y1, w, h)
    # cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    # (x_c, y_c, w, h)
    x1 = int(x - w / 2.0)
    x2 = int(x + w / 2.0)
    y1 = int(y - h / 2.0)
    y2 = int(y + h / 2.0)
    print(x1, y1, x2, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    # x1, y1, x2, y2 = bbox
    # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


# 加载图像
image = cv2.imread(
    '/Users/messiz/Documents/visual grounding/codes/TransVG/test/a10dea57-90f876f4-c66af250-6fb45322-6ef88ddc.jpg')
print(image.shape)

# 假设有一个边界框的坐标（左上角点的坐标和宽高）
bbox = (1485,
        1504,
        673,
        655)

# 在图像上绘制边界框
draw_bbox(image, bbox)

# 显示图像
cv2.imshow('Image with Bbox', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ms-cxr bbox标注格式是（x1, y1, w, h）
