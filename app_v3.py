import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- 1. قاموس اللغات ---
LANGUAGES = {
    "English": {
        "title": "RetinaCheck AI: Early Diagnosis System",
        "sub": "AI-powered system to assist doctors in detecting retinal diseases.",
        "upload": "Please upload a fundus image...",
        "orig": "Original Image",
        "proc": "Medical Filter (Ben Graham)",
        "detecting": "AI analyzing and processing...",
        "safe": "Healthy",
        "danger": "Infected",
        "conf": "Confidence Level",
        "note": "Note: This is a technical analysis tool, the final decision belongs to the specialist.",
        "student": "Student: SALEH SABRI ALTAMIMI",
        "supervisor": "Supervisor: Prof. Dr. Yusuf ÖZCAN",
        "auto_msg": "System detected: ",
        "steps": "Processing Steps",
        # Processing step names
        "step_1": "Convert image to NumPy array",
        "step_2": "Convert from RGB to BGR",
        "step_3": "Check: Already processed image, resizing only",
        "step_4": "Resize image to 512x512",
        "step_5": "Check: Raw image, full processing will be applied",
        "step_6": "Convert to grayscale",
        "step_7": "Crop image to remove black edges",
        "step_8": "Convert to LAB color space",
        "step_9": "Split L, A, B channels",
        "step_10": "Apply CLAHE to L channel for contrast enhancement",
        "step_11": "Merge channels again",
        "step_12": "Convert to BGR",
        "step_13": "Apply Ben Graham filter (addWeighted with Gaussian Blur)",
        "step_14": "Apply circular mask to focus on center area"
    },
    "Türkçe": {
        "title": "RetinaCheck AI: Erken Teşhis Sistemi",
        "sub": "Doktorlara retina hastalıklarını tespit etmede yardımcı olan yapay zeka sistemi.",
        "upload": "Lütfen fundus görüntüsünü yükleyin...",
        "orig": "Orijinal Görüntü",
        "proc": "Tıbbi Filtre (Ben Graham)",
        "detecting": "Yapay zeka analiz ediyor ve işliyor...",
        "safe": "Sağlıklı",
        "danger": "Enfekte",
        "conf": "Güven Seviyesi",
        "note": "Not: Bu teknik bir analiz aracıdır, nihai karar uzmana aittir.",
        "student": "Öğrenci: SALEH SABRI ALTAMIMI",
        "supervisor": "Danışman: Prof. Dr. Yusuf ÖZCAN",
        "auto_msg": "Sistem algıladı: ",
        "steps": "İşleme Adımları",
        # Processing step names
        "step_1": "Görüntüyü NumPy dizisine dönüştür",
        "step_2": "RGB'den BGR'ye dönüştür",
        "step_3": "Kontrol: Önceden işlenmiş görüntü, sadece yeniden boyutlandırma",
        "step_4": "Görüntüyü 512x512 olarak yeniden boyutlandır",
        "step_5": "Kontrol: Ham görüntü, tam işleme uygulanacak",
        "step_6": "Gri tonlamaya dönüştür",
        "step_7": "Siyah kenarları kaldırmak için görüntüyü kırp",
        "step_8": "LAB renk uzayına dönüştür",
        "step_9": "L, A, B kanallarını ayır",
        "step_10": "Kontrast geliştirme için L kanalına CLAHE uygula",
        "step_11": "Kanalları tekrar birleştir",
        "step_12": "BGR'ye dönüştür",
        "step_13": "Ben Graham filtresi uygula (Gaussian Blur ile addWeighted)",
        "step_14": "Merkez alana odaklanmak için dairesel maske uygula"
    },
    "العربية": {
        "title": "RetinaCheck AI: نظام التشخيص المبكر",
        "sub": "نظام يعتمد على الذكاء الاصطناعي لمساعدة الأطباء في كشف أمراض الشبكية.",
        "upload": "يرجى رفع صورة قاع العين...",
        "orig": "الصورة الأصلية",
        "proc": "الفلتر الطبي (Ben Graham)",
        "detecting": "جاري التحليل والمعالجة الذكية...",
        "safe": "سليم",
        "danger": "مصاب",
        "conf": "نسبة التأكد",
        "note": "تنبيه: هذه أداة تحليل تقنية، القرار النهائي يعود للطبيب المختص.",
        "student": "الطالب: صالح صبري التميمي",
        "supervisor": "المشرف: Prof. Dr. Yusuf ÖZCAN",
        "auto_msg": "النظام اكتشف: ",
        "steps": "خطوات المعالجة",
        # Processing step names
        "step_1": "تحويل الصورة إلى مصفوفة NumPy",
        "step_2": "تحويل من RGB إلى BGR",
        "step_3": "فحص: الصورة معالجة مسبقاً، سيتم توحيد الحجم فقط",
        "step_4": "تغيير حجم الصورة إلى 512x512",
        "step_5": "فحص: الصورة خام، سيتم المعالجة الكاملة",
        "step_6": "تحويل إلى صورة رمادية",
        "step_7": "قص الصورة لإزالة الحواف السوداء",
        "step_8": "تحويل إلى فضاء LAB",
        "step_9": "فصل القنوات L, A, B",
        "step_10": "تطبيق CLAHE على قناة L لتحسين التباين",
        "step_11": "دمج القنوات مرة أخرى",
        "step_12": "تحويل إلى BGR",
        "step_13": "تطبيق فلتر Ben Graham (addWeighted مع Gaussian Blur)",
        "step_14": "تطبيق قناع دائري لتركيز على المنطقة المركزية"
    }
}

# --- 2. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="RetinaCheck AI", page_icon="👁️", layout="wide")

# اختيار اللغة
selected_lang = st.sidebar.selectbox("🌐 Language / Dil / اللغة", ["English", "Türkçe", "العربية"])
L = LANGUAGES[selected_lang]

# تصميم CSS مخصص
st.markdown(f"""
    <style>
    .main {{ background-color: #f8f9fa; }}
    .stAlert {{ border-radius: 10px; }}
    .footer {{ position: fixed; bottom: 0; width: 100%; text-align: center; color: #6c757d; font-size: 14px; padding: 10px; background: white; }}
    {"div[data-testid='stBlock'] {direction: rtl; text-align: right;}" if selected_lang == "العربية" else ""}
    </style>
    """, unsafe_allow_html=True)

# --- 3. المحرك الذكي (Smart Preprocessor) ---
def is_already_processed(img):
    std_dev = np.std(img)
    return std_dev < 35

def smart_preprocess(image_pil, size=512, lang_dict=None):
    if lang_dict is None:
        lang_dict = L
    
    steps = []
    
    img = np.array(image_pil)
    steps.append({
        "name": lang_dict["step_1"],
        "image": img.copy()
    })
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    steps.append({
        "name": lang_dict["step_2"],
        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    })
    
    if is_already_processed(img):
        status = "Already Processed / İşlenmiş / معالجة مسبقاً"
        steps.append({
            "name": lang_dict["step_3"],
            "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        })
        img = cv2.resize(img, (size, size))
        steps.append({
            "name": lang_dict["step_4"],
            "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        })
        return img, status, steps
    
    status = "Raw Image / Ham Görüntü / صورة خام"
    steps.append({
        "name": lang_dict["step_5"],
        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    })
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append({
        "name": lang_dict["step_6"],
        "image": cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    })
    
    mask = gray > 10
    if mask.any(): 
        img = img[np.ix_(mask.any(1), mask.any(0))]
        steps.append({
            "name": lang_dict["step_7"],
            "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        })
    
    img = cv2.resize(img, (size, size))
    steps.append({
        "name": lang_dict["step_4"],
        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    })
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    steps.append({
        "name": lang_dict["step_8"],
        "image": cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    })
    
    l, a, b = cv2.split(lab)
    steps.append({
        "name": lang_dict["step_9"],
        "image": cv2.cvtColor(cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    })
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    steps.append({
        "name": lang_dict["step_10"],
        "image": cv2.cvtColor(cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    })
    
    img = cv2.merge((l, a, b))
    steps.append({
        "name": lang_dict["step_11"],
        "image": cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    })
    
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    steps.append({
        "name": lang_dict["step_12"],
        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    })
    
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), size/30), -4, 128)
    steps.append({
        "name": lang_dict["step_13"],
        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    })
    
    mask_circ = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask_circ, (size//2, size//2), int(size/2.1), 1, -1)
    img = cv2.bitwise_and(img, img, mask=mask_circ)
    steps.append({
        "name": lang_dict["step_14"],
        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    })
    
    return img, status, steps

# --- 4. واجهة العرض الرئيسية ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://cdn.pau.edu.tr/BIYS/logo/PAUlogoTR.png", width=120)
with col_title:
    st.title(L["title"])
    st.write(f"**{L['student']}** | **{L['supervisor']}**")

st.write(L["sub"])

# تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# رفع الصورة
uploaded_file = st.file_uploader(L["upload"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # المعالجة والذكاء التلقائي
    with st.spinner(L["detecting"]):
        processed_img, detection_status, steps = smart_preprocess(original_image, lang_dict=L)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {L['orig']}")
            st.image(original_image, use_container_width=True)
        with c2:
            st.markdown(f"### {L['proc']}")
            display_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(display_rgb, use_container_width=True)
            st.caption(f"{L['auto_msg']} {detection_status}")

        # عرض خطوات المعالجة مع الصور
        st.markdown(f"### {L['steps']}")
        
        # إنشء أعمدة لعرض الصور بشكل جانبي
        cols_per_row = 2
        for i in range(0, len(steps), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                step_index = i + j
                if step_index < len(steps):
                    step = steps[step_index]
                    with col:
                        st.image(step["image"], use_container_width=True)
                        st.caption(f"**{step_index + 1}. {step['name']}**")

        # التنبؤ
        results = model.predict(processed_img, imgsz=512)
        probs = results[0].probs
        prediction = "Infected" if probs.top1 == 1 else "Healthy"
        confidence = probs.top1conf.item()

        # النتيجة النهائية
        st.divider()
        if prediction == "Infected":
            st.error(f"## {L['danger']} ({confidence:.2%})")
        else:
            st.success(f"## {L['safe']} ({confidence:.2%})")
        
        st.progress(confidence)

st.markdown(f"<div class='footer'>{L['note']}</div>", unsafe_allow_html=True)
