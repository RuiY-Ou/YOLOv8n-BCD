from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"F:\deeplearning\ultralytics-8.3.163\ultralytics\cfg\models\v8\yolov8n-BiFPN+CA+Dy_Sample.yaml")                #可以修改所要加载的模型
    model.train(
        data=r"F:\deeplearning\ultralytics-8.3.163\ultralytics\cfg\datasets\BDD100K_night.yaml",             #打算训练的数据集  现在打算训练 coco8 这个数据集
        epochs=600,                               #打算训练的轮数
        imgsz=640,                                #图片尺寸 32的倍数 越小训练越快 根据图片尺寸修改 大改大 小改小 默认640
        batch=-1,                                 #批量 一次喂几张图片进行训练 所有批次训练完为训练完一轮 -1->自动设置最优batch
        cache=False,                              #缓存 False对应"ram" 即在训练前YOLO提前把图片加载进内存并缩放好 图片尺寸很大最有用
        workers=2,                                #工作者，暗指进程 多几个打包过程（多几个打工仔）
        patience=200,                              # 如果20轮没有改善则停止

        cos_lr=True,  # 余弦退火
        optimizer='AdamW',  # AdamW优化器/SGD优化器
        lr0=0.01,     #原0.01
        lrf=0.001,  # 更低的学习率衰减
        weight_decay=0.05,  # 权重衰减
        hsv_h=0.015,  # 色相
        hsv_s=0.7,  # 饱和度
        hsv_v=0.4,  # 明度（关键）
    )


