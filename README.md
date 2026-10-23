建筑物损伤分割与检测数据集，5925张，voc，yolo格式分割与检测标注信息
4类，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/586b7a819fb74fe9aecd452f723ad0a8.png)
标注数量：
钢筋外露 Rebar-Exposure 1281
腐蚀 Corrosion 6843
泛碱 Efflorescence 2297
开裂 Cracked 3471
图像数量 image num 5925![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f8af542054c04fd98a3ecbfbf5cef2a5.png)
# 建筑物损伤分割与检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2f7e54b940d1420eaaddef6b2ea6294c.png)

## 数据集总览表![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99d34d2e781a4959b1d13f4a874594df.png)

|项目|详情|
| ---- | ---- |
|数据集名称|建筑物损伤分割与检测数据集|
|图像总数|5925张|
|标注格式|YOLO、VOC、JSON（同时支持目标检测+实例分割）|
|类别总数|4类|
|训练集|4740张|
|验证集|711张|
|测试集|474张|

## 类别统计表
|英文标签|中文标签|含该类图像数|标注总数|
| ---- | ---- | ---- | ---- |
|Rebar-Exposure|钢筋外露|1281|1281|
|Corrosion|腐蚀|6843|6843|
|Efflorescence|泛碱|2297|2297|
|Cracked|开裂|3471|3471|

类别：数据集包含钢筋外露、腐蚀、泛碱、开裂4类建筑物表面损伤。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ec0702bd07ed4df694dd8ca8f6b46d36.png)

## 关键词
建筑病害检测、建筑物表面损伤、钢筋外露、墙体开裂、混凝土腐蚀、泛碱、实例分割、YOLOv11、结构健康监测、无人机建筑巡检
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c8f9f990b22452db7ca861b81564766.png)

## 应用场景
1. 建筑结构健康监测，无人机航拍/近距离拍照识别混凝土墙体开裂、钢筋外露、腐蚀、泛碱等病害，辅助房屋安全鉴定。
2. 老旧房屋、桥梁、隧道外立面自动巡检，替代人工肉眼检测，降低高空作业风险。
3. 土木工程、机器视觉方向课题、毕业设计、算法模型研发；可同时做目标检测与实例分割任务。
4. 建筑运维平台，自动统计各类病害面积，评估建筑物损伤等级，为维修方案提供数据支撑。

## YOLOv11 训练代码（实例分割）
### 1. 数据集配置文件 `building_damage.yaml`
```yaml
path: "./building_damage_dataset"
train: images/train
val: images/val
test: images/test

names:
  0: Rebar-Exposure
  1: Corrosion
  2: Efflorescence
  3: Cracked
```

### 2. 训练脚本 `train_building_seg.py`
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11分割预训练权重
    model = YOLO("yolo11n-seg.pt")
    train_result = model.train(
        data="building_damage.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="building_damage_result",
        name="yolo11_building_damage",
        augment=True,
        conf=0.25
    )
```

### 3. 推理脚本
```python
from ultralytics import YOLO
model = YOLO("./building_damage_result/yolo11_building_damage/weights/best.pt")
results = model.predict("test_building.jpg", save=True)
```

### 环境安装命令
```bash
pip install ultralytics torch opencv-python
```

## PyQt5简易可视化界面代码
```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class BuildingDamageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("建筑物损伤分割检测系统")
        self.setGeometry(100, 100, 1000, 700)
        self.model = YOLO("./building_damage_result/yolo11_building_damage/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        self.img_label = QLabel("图像预览区域", self)
        self.img_label.setGeometry(20, 20, 750, 600)
        self.img_label.setStyleSheet("border:1px solid #666;")
        self.img_label.setAlignment(Qt.AlignCenter)

        self.btn_detect = QPushButton("图片损伤检测", self)
        self.btn_detect.setGeometry(800, 50, 160, 40)
        self.btn_detect.clicked.connect(self.detect_image)

    def detect_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择建筑物图片", "", "图片(*.jpg *.png *.bmp)")
        if not file_path:
            return
        result = self.model(file_path)[0]
        img_res = result.plot()
        h, w, c = img_res.shape
        qt_img = QImage(img_res.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.img_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.img_label.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BuildingDamageWindow()
    win.show()
    sys.exit(app.exec_())
```

