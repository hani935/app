from ultralytics import YOLO

# مدل سبک برای شروع آموزش
model = YOLO("yolo26n.pt")

# آموزش مدل تخصصی تجهیزات ایمنی
model.train(
    data="https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
    epochs=30,
    imgsz=416,
    batch=2,
    device="cpu",
    workers=0,
    patience=8,
    project="SafeMind",
    name="PPE_Model"
)