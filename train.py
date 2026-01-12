from ultralytics import YOLO


def main():
    # ===================== CONFIG =====================
    MODEL_WEIGHTS = "Weight/yolo12x.pt"   # or yolov12m.pt
    DATA_YAML = "data.yaml"

    IMG_SIZE = 640          # IMPORTANT for small buds
    EPOCHS = 200
    BATCH = 8                # adjust based on GPU
    DEVICE = 'cpu'               # 0 = GPU, 'cpu' if no GPU

    PROJECT = "tea_plucking"
    NAME = "yolov12_bud_leaf"

    # ===================== LOAD MODEL =================
    model = YOLO(MODEL_WEIGHTS)

    # ===================== TRAIN ======================
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,

        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,

        patience=30,
        warmup_epochs=5,

        close_mosaic=10,     # disable mosaic at end
        mosaic=0.3,          # reduce mosaic (buds are small)

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,
        translate=0.1,
        scale=0.3,
        shear=0.0,

        flipud=0.0,
        fliplr=0.5,

        workers=8,
        rect=True,

        project=PROJECT,
        name=NAME,
        exist_ok=True,

        pretrained=True,
        verbose=True
    )

    # ===================== VALIDATE ===================
    model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=DEVICE
    )

    print("✅ Training & validation completed successfully!")


if __name__ == "__main__":
    main()
