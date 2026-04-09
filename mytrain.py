from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"F:\deeplearning\ultralytics-8.3.163\ultralytics\cfg\models\v8\yolov8n-BiFPN+CA+Dy_Sample.yaml")               # model path
    model.train(
        data=r"",                                                                                                                # dataset path     
        epochs=600,                          
        imgsz=640,                          
        batch=-1,                           
        cache=False,                         
        workers=2,                            
        patience=200,                           

        cos_lr=True,  
        optimizer='AdamW', 
        lr0=0.01,     
        lrf=0.001, 
        weight_decay=0.05,  
        hsv_h=0.015, 
        hsv_s=0.7,
        hsv_v=0.4
    )


