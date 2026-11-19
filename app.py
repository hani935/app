import tempfile
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2
from datetime import datetime
from pathlib import Path
import numpy as np

# -----------------------------
# تنظیمات برنامه
# -----------------------------
st.set_page_config(
    page_title="SafeMind AI",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 SafeMind AI")
st.subheader("سامانه هوشمند پایش و ارزیابی اولیه ایمنی کارخانه")

st.info(
    "این سامانه ابزار کمکی برای بررسی تصاویر است "
    "و جایگزین نظارت انسانی و استانداردهای ایمنی نیست."
)

# -----------------------------
# بارگذاری مدل
# -----------------------------
@st.cache_resource
def load_model():
    general_model = YOLO("yolov8n.pt")
    ppe_model = YOLO("models/best.pt")
    return general_model, ppe_model


try:
    model, ppe_model = load_model()
    st.success("هر دو مدل هوش مصنوعی آماده هستند")

except Exception as e:
    st.error(f"خطا در بارگذاری مدل: {e}")
    st.stop()


# -----------------------------
# دسته‌بندی اشیا
# -----------------------------
OBSTACLE_CLASSES = {
    "chair",
    "bench",
    "box",
    "bicycle",
    "motorcycle",
    "forklift"
}


def analyze_objects(results):
    names = results.names
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = names[class_id]

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        detections.append({
            "name": name,
            "confidence": confidence,
            "box": (x1, y1, x2, y2)
        })

    return detections


# -----------------------------
# بررسی نزدیکی اشیا
# -----------------------------
def is_near(person, obj):

    px1, py1, px2, py2 = person["box"]
    ox1, oy1, ox2, oy2 = obj["box"]

    person_center = (
        (px1 + px2) / 2,
        (py1 + py2) / 2
    )

    obj_center = (
        (ox1 + ox2) / 2,
        (oy1 + oy2) / 2
    )

    distance = np.sqrt(
        (person_center[0] - obj_center[0]) ** 2 +
        (person_center[1] - obj_center[1]) ** 2
    )

    person_height = max(py2 - py1, 1)

    return distance < person_height * 0.8


# -----------------------------
# حذف تشخیص‌های تکراری انسان
# -----------------------------
def remove_duplicate_persons(persons):

    unique_persons = []

    for person in persons:

        x1, y1, x2, y2 = person["box"]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        duplicate = False

        for existing in unique_persons:

            a1, b1, a2, b2 = existing["box"]

            other_x = (a1 + a2) / 2
            other_y = (b1 + b2) / 2

            distance = np.sqrt(
                (center_x - other_x) ** 2 +
                (center_y - other_y) ** 2
            )

            if distance < max(
                y2 - y1,
                b2 - b1
            ) * 0.5:

                duplicate = True
                break

        if not duplicate:
            unique_persons.append(person)

    return unique_persons


# -----------------------------
# بارگذاری تصویر
# -----------------------------

# -----------------------------
# بارگذاری تصویر یا ویدئو
# -----------------------------
uploaded_file = st.file_uploader(
    "تصویر یا ویدئوی محیط کارخانه را انتخاب کن",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

report = None

if uploaded_file:

    file_suffix = Path(uploaded_file.name).suffix.lower()
    is_video = file_suffix in [".mp4", ".avi", ".mov"]

    try:

        # =================================
        # حالت اول: تحلیل تصویر
        # =================================
        if not is_video:

            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)

            st.image(
                image,
                caption="تصویر ورودی",
                use_container_width=True
            )

            with st.spinner("در حال تحلیل تصویر..."):

                general_results = model.predict(
                    source=image_array,
                    conf=0.10,
                    iou=0.35,
                    imgsz=1280,
                    max_det=1000,
                    classes=[i for i in range(80) if i != 26],
                    verbose=False
                )[0]

                ppe_results = ppe_model.predict(
                    source=image_array,
                    conf=0.10,
                    imgsz=960,
                    max_det=1000,
                    verbose=False
                )[0]

            general_detections = analyze_objects(general_results)
            ppe_detections = analyze_objects(ppe_results)

            detections = general_detections + ppe_detections

            persons = [
                item for item in detections
                if item["name"].lower() == "person"
            ]

            unique_persons = remove_duplicate_persons(persons)

            obstacles = [
                item for item in general_detections
                if item["name"].lower() in OBSTACLE_CLASSES
            ]

            alerts = []

            for person in unique_persons:
                for obj in obstacles:
                    if is_near(person, obj):
                        alerts.append(
                            f"فردی در نزدیکی شیء قابل بررسی "
                            f"({obj['name']}) دیده شده است."
                        )

            if not unique_persons:
                alerts.append("هیچ انسانی در تصویر شناسایی نشد.")

            if not obstacles:
                alerts.append(
                    "مانع قابل شناسایی از فهرست تعریف‌شده پیدا نشد."
                )

            # گزارش PPE
            st.subheader("🦺 گزارش تجهیزات ایمنی")

            ppe_classes = {
                "helmet": "کلاه ایمنی",
                "vest": "جلیقه ایمنی",
                "boots": "کفش ایمنی",
                "gloves": "دستکش ایمنی",
                "goggles": "عینک ایمنی",
                "no_helmet": "مورد نیازمند بررسی کلاه ایمنی",
                "no_vest": "مورد نیازمند بررسی جلیقه ایمنی",
                "no_boots": "مورد نیازمند بررسی کفش ایمنی",
                "no_gloves": "مورد نیازمند بررسی دستکش ایمنی",
                "no_goggle": "مورد نیازمند بررسی عینک ایمنی"
            }

            for class_name, persian_name in ppe_classes.items():
                count = sum(
                    1 for item in ppe_detections
                    if item["name"].lower() == class_name
                )

                if count > 0:
                    st.write(
                        f"🔎 {persian_name}: {count} مورد تشخیص داده شد"
                    )

            # نمایش مدل عمومی
            st.subheader("📷 نتیجه تشخیص مدل عمومی")

            general_annotated = general_results.plot()
            general_annotated = cv2.cvtColor(
                general_annotated,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                general_annotated,
                use_container_width=True
            )

            # نمایش مدل PPE
            st.subheader("🦺 نتیجه تشخیص تجهیزات ایمنی")

            ppe_annotated = ppe_results.plot()
            ppe_annotated = cv2.cvtColor(
                ppe_annotated,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                ppe_annotated,
                use_container_width=True
            )

            # آمار
            st.subheader("📊 آمار تحلیل")

            col1, col2, col3 = st.columns(3)

            col1.metric("تعداد افراد یکتا",len(person_track_ids))
            col2.metric("تعداد اشیا", len(detections))
            col3.metric("تعداد هشدارها", len(alerts))
            ppe_totals = {}

            for statuses in person_ppe_status.values():
                for name in statuses:
                    ppe_totals[name] = ppe_totals.get(name, 0) + 1

            # هشدارها
            st.subheader("🚨 هشدارهای هوشمند")

            for alert in alerts:
                st.warning(alert)

            if unique_persons and not alerts:
                st.success("هشدار اولیه‌ای ثبت نشد.")

            # جدول
            st.subheader("🔍 فهرست اشیای شناسایی‌شده")

            object_data = [
                {
                    "نام شیء": item["name"],
                    "میزان اطمینان": round(
                        item["confidence"] * 100, 2
                    )
                }
                for item in detections
            ]

            if object_data:
                st.dataframe(
                    pd.DataFrame(object_data),
                    use_container_width=True
                )

            report = {
                "نوع ورودی": "تصویر",
                "زمان": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "تعداد انسان": len(unique_persons),
                "تعداد اشیا": len(detections),
                "تعداد هشدار": len(alerts),
                "هشدارها": " | ".join(alerts)
            }

        # =================================
        # حالت دوم: تحلیل ویدئو
        # =================================
        else:

            st.subheader("🎬 تحلیل ویدئوی محیط کارخانه")

            st.video(uploaded_file.getvalue(),format="video/mp4")

            st.info(
                "برای کاهش زمان پردازش، یک فریم از هر ۵ فریم بررسی می‌شود."
            )

            # ذخیره موقت ویدئوی ورودی
            with tempfile.NamedTemporaryFile(
                suffix=file_suffix,
                delete=False
            ) as temp_input:

                temp_input.write(uploaded_file.getbuffer())
                input_path = temp_input.name

            output_file = tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            )

            output_path = output_file.name
            output_file.close()

            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
             st.error("OpenCV نتوانست فایل ویدئو را باز کند.")
             st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                st.error("هیچ فریمی از ویدئو خوانده نشد.")
                cap.release()
                st.stop()

            st.success(f"ویدئو باز شد؛ تعداد فریم‌ها: {total_frames}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25

            stride = 5
            output_fps = max(fps / stride, 1)

            writer = None
            frame_index = 0
            processed_frames = 0

            person_track_ids = set()
            total_objects = 0
            total_alerts = 0

            ppe_totals = {}
            person_ppe_status={}
            video_alerts = []

            progress = st.progress(0)
            status = st.empty()

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            try:

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_index += 1

                    # بررسی یک فریم از هر پنج فریم
                    if frame_index % stride != 0:
                        continue

                    general_results = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=0.25,
                        imgsz=640,
                        max_det=300,
                        classes=[0] + [i for i in range(80) if i not in [0, 26]],
                        verbose=False
                    )[0]
                    ppe_results = ppe_model.predict(
                        source=frame,
                        conf=0.10,
                        imgsz=640,
                        max_det=300,
                        verbose=False
                    )[0]

                    general_detections = analyze_objects(
                        general_results
                    )

                    ppe_detections = analyze_objects(
                        ppe_results
                    )

                    detections = (
                        general_detections + ppe_detections
                    )

                    persons = [
                        item for item in detections
                        if item["name"].lower() == "person"
                    ]

                    unique_persons = remove_duplicate_persons(
                        persons
                    )

                    obstacles = [
                        item for item in general_detections
                        if item["name"].lower() in OBSTACLE_CLASSES
                    ]

                    if general_results.boxes.id is not None:
                        for class_id, track_id in zip(
                          general_results.boxes.cls.cpu().tolist(),
                            general_results.boxes.id.cpu().tolist()
                        ):
                            if int(class_id) == 0:
                                person_track_ids.add(int(track_id))
                    total_objects += len(detections)

                    if general_results.boxes.id is not None:

                        person_boxes = zip(
                        general_results.boxes.xyxy.cpu().tolist(),
                        general_results.boxes.cls.cpu().tolist(),
                        general_results.boxes.id.cpu().tolist())
                        for box, class_id, track_id in person_boxes:

                            if int(class_id) != 0:
                                continue

                            track_id = int(track_id)
                            x1, y1, x2, y2 = box

                            person_ppe_status.setdefault(track_id, set())
                            for item in ppe_detections:

                                name = item["name"].lower()

                                if name == "person":
                                    continue
                                ix1, iy1, ix2, iy2 = item["box"]

                                center_x = (ix1 + ix2) / 2
                                center_y = (iy1 + iy2) / 2

                                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                                    person_ppe_status[track_id].add(name)

                    for person in unique_persons:
                        for obj in obstacles:
                            if is_near(person, obj):
                                total_alerts += 1
                                video_alerts.append(
                                    f"فردی در نزدیکی {obj['name']} "
                                    f"در فریم {frame_index} دیده شد."
                                )

                    # ساخت ویدئوی خروجی با کادرهای PPE
                    annotated = general_results.plot()
                    annotated = ppe_results.plot(img=annotated)

                    if writer is None:
                        height, width = annotated.shape[:2]

                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            output_path,
                            fourcc,
                            output_fps,
                            (width, height)
                        )

                    writer.write(annotated)

                    processed_frames += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_index / total_frames, 1.0)
                        )

                    status.write(
                        f"فریم بررسی‌شده: {processed_frames}"
                    )

            finally:
                cap.release()

                if writer is not None:
                    writer.release()

                Path(input_path).unlink(missing_ok=True)

            progress.progress(1.0)

            if processed_frames == 0:
                st.error("فریم قابل پردازشی در ویدئو پیدا نشد.")

                Path(output_path).unlink(missing_ok=True)

            else:

                st.success("تحلیل ویدئو به پایان رسید.")

                st.subheader("🎥 ویدئوی پردازش‌شده")
                st.video(output_path)

                st.subheader("📊 آمار تحلیل ویدئو")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "مجموع تشخیص انسان",
                    total_persons
                )

                col2.metric(
                    "مجموع تشخیص اشیا",
                    total_objects
                )

                col3.metric(
                    "تعداد هشدارها",
                    total_alerts
                )

                st.subheader("🦺 گزارش تجهیزات ایمنی")

                ppe_names = {
                    "helmet": "کلاه ایمنی",
                    "vest": "جلیقه ایمنی",
                    "boots": "کفش ایمنی",
                    "gloves": "دستکش ایمنی",
                    "goggles": "عینک ایمنی",
                    "no_helmet": "مورد نیازمند بررسی کلاه",
                    "no_vest": "مورد نیازمند بررسی جلیقه",
                    "no_boots": "مورد نیازمند بررسی کفش",
                    "no_gloves": "مورد نیازمند بررسی دستکش",
                    "no_goggle": "مورد نیازمند بررسی عینک"
                }

                for class_name, persian_name in ppe_names.items():
                    count = ppe_totals.get(class_name, 0)

                    if count > 0:
                        st.write(
                            f"🔎 {persian_name}: {count} نفر"
                        )

                st.caption(
                    "این اعداد مجموع تشخیص‌ها در فریم‌های نمونه‌برداری‌شده "
                    "هستند، نه تعداد افراد یکتا."
                )

                st.subheader("🚨 هشدارهای ویدئو")

                if video_alerts:
                    for alert in video_alerts[:30]:
                        st.warning(alert)
                else:
                    st.success(
                        "هشدار اولیه‌ای ثبت نشد."
                    )

                report = {
                    "نوع ورودی": "ویدئو",
                    "زمان": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "تعداد انسان": total_persons,
                    "تعداد اشیا": total_objects,
                    "تعداد هشدار": total_alerts,
                    "هشدارها": " | ".join(video_alerts)
                }

    except Exception as e:
        st.error(f"خطا هنگام تحلیل: {e}")

# =================================
# ذخیره و دانلود گزارش
# =================================

if report is not None:

    try:

        report_file = Path("safemind_reports.csv")

        report_df = pd.DataFrame([report])

        if report_file.exists():

            old_reports = pd.read_csv(report_file)

            report_df = pd.concat(
                [old_reports, report_df],
                ignore_index=True
            )

        report_df.to_csv(
            report_file,
            index=False,
            encoding="utf-8-sig"
        )

        st.success("گزارش تحلیل ذخیره شد.")

        csv_data = report_df.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="⬇️ دانلود گزارش‌ها",
            data=csv_data,
            file_name="safemind_reports.csv",
            mime="text/csv",
            key="download_safemind_reports"
        )

    except Exception as e:
        st.error(f"خطا در ذخیره گزارش: {e}")

else:
    if not uploaded_file:
        st.info("برای شروع تحلیل، یک تصویر یا ویدئو انتخاب کن.")
