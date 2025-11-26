import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import time
import pyttsx3
import threading
from fpdf import FPDF
import datetime
import pandas as pd
import os

# ==========================================
# ⚙️ CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="ISRO Safety Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 ENHANCED NEON THEME
st.markdown("""
    <style>
    /* Deep Space Background */
    .stApp {
        background: linear-gradient(135deg, #0A0E17 0%, #161B22 100%);
        color: #E8E8E8;
    }
    
    /* Sidebar with gradient */
    .stSidebar {
        background: linear-gradient(180deg, #1A1F2C 0%, #161B22 100%);
        border-right: 2px solid #00ADB5;
    }
    
    /* Enhanced Neon Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00F5FF !important;
        text-shadow: 0 0 15px rgba(0, 245, 255, 0.7);
        background: linear-gradient(90deg, #00F5FF, #00ADB5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Cards with glass morphism effect */
    .card {
        background: rgba(25, 30, 45, 0.7);
        border-radius: 15px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced Metric Values */
    div[data-testid="stMetricValue"] {
        color: #00FFC2;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 1.5rem;
        text-shadow: 0 0 10px #00FFC2;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8892B0 !important;
        font-size: 0.9rem;
    }
    
    /* Enhanced Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00ADB5, #007BFF);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 173, 181, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 173, 181, 0.5);
    }
    
    /* Warning Box */
    .warning-box {
        border: 2px solid #FF2E63;
        background: linear-gradient(45deg, rgba(255, 46, 99, 0.1), rgba(255, 46, 99, 0.05));
        padding: 15px;
        border-radius: 10px;
        color: #FF2E63;
        font-weight: bold;
        text-align: center;
        animation: pulse 1.5s infinite;
        box-shadow: 0 0 20px rgba(255, 46, 99, 0.3);
    }
    
    /* Success Box */
    .success-box {
        border: 2px solid #00FFC2;
        background: linear-gradient(45deg, rgba(0, 255, 194, 0.1), rgba(0, 255, 194, 0.05));
        padding: 15px;
        border-radius: 10px;
        color: #00FFC2;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 194, 0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(255, 46, 99, 0.3); }
        50% { transform: scale(1.02); box-shadow: 0 0 30px rgba(255, 46, 99, 0.5); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(255, 46, 99, 0.3); }
    }
    
    /* Custom progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ADB5, #007BFF);
    }
    
    /* Safety gauge */
    .gauge-container {
        background: rgba(25, 30, 45, 0.8);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0, 245, 255, 0.3);
        text-align: center;
    }
    
    .gauge-value {
        font-size: 3rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    
    .gauge-label {
        font-size: 1.2rem;
        color: #8892B0;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🧠 AI ENGINE (CACHED)
# ==========================================
@st.cache_resource
def load_model(model_path):
    try:
        with st.spinner('🔄 Loading AI Model...'):
            # Simulate loading time for effect
            time.sleep(1)
            return YOLO(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# ==========================================
# 🔥 HEATMAP GENERATOR (ROBUST FIX)
# ==========================================
def generate_heatmap_overlay(image, detections):
    # 1. Convert PIL Image to NumPy (Fixes 'shape' error)
    image = np.array(image)
    
    # 2. Ensure RGB (Fixes color glitches)
    if image.ndim == 2: # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
    # 3. Create blank mask
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    # 4. Draw Hotspots
    # We check if detections exist and are not empty
    if detections is not None and len(detections) > 0:
        for box in detections:
            # CRITICAL FIX: Move tensor to CPU and convert to int list
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            # Add "heat" to center
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max(20, int(min(x2-x1, y2-y1) / 2))
            
            # Draw Gaussian blob (Brightness = 1.0)
            cv2.circle(heatmap, (center_x, center_y), radius, (1.0), -1)
    
        # 5. Blur and Colorize
        heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)
        
        # Avoid division by zero if heatmap is empty
        if np.max(heatmap) > 0:
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        
        heatmap = np.uint8(heatmap)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 6. Blend (Original + Heatmap)
        # Convert image to BGR for OpenCV blending
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_img, 0.4, 0)
        
        # Return RGB for Streamlit
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
    else:
        # If no detections, just return original image dimmed slightly
        return image

# ==========================================
# 🔊 VOICE ALERT SYSTEM (THREADED)
# ==========================================
def speak_warning(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.8)
            engine.say(text)
            engine.runAndWait()
        except:
            pass 
    thread = threading.Thread(target=_speak)
    thread.daemon = True
    thread.start()

# ==========================================
# 📄 PROFESSIONAL REPORT GENERATOR
# ==========================================
def generate_pdf(score, items, timestamp, log_data):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(0, 173, 181)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "ISRO SAFETY AUDIT REPORT", 0, 1, "C")
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, "Bharatiya Antariksha Station", 0, 1, "C")
    
    # Mission Details
    pdf.set_y(50)
    pdf.set_draw_color(0, 173, 181)
    pdf.set_line_width(0.5)
    pdf.line(10, 50, 200, 50)
    
    pdf.set_y(60)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Mission ID:", 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.cell(100, 10, "GAGANYAAN-SAFETY-01", 0, 1)
    
    pdf.cell(40, 10, "Timestamp:", 0, 0)
    pdf.cell(100, 10, f"{timestamp}", 0, 1)
    
    pdf.cell(40, 10, "Safety Score:", 0, 0)
    if score < 50:
        pdf.set_text_color(255, 0, 0)
    elif score < 80:
        pdf.set_text_color(255, 165, 0)
    else:
        pdf.set_text_color(0, 128, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(100, 10, f"{score}%", 0, 1)
    
    # Safety Gauge
    pdf.set_y(100)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Safety Assessment", 0, 1)
    
    # Simple gauge representation
    pdf.rect(10, 115, 190, 15)
    fill_width = (score / 100) * 190
    pdf.set_fill_color(0, 173, 181)
    pdf.rect(10, 115, fill_width, 15, 'F')
    
    # Inventory Table
    pdf.set_y(140)
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, "Equipment Name", 1, 0, "C", True)
    pdf.cell(40, 10, "Quantity", 1, 0, "C", True)
    pdf.cell(50, 10, "Status", 1, 1, "C", True)
    
    pdf.set_font("Arial", "", 11)
    fill = False
    if items:
        for item, count in items.items():
            pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
            # Encode/Decode to clean strings
            clean_item = str(item).encode('latin-1', 'ignore').decode('latin-1')
            pdf.cell(100, 10, f"  {clean_item}", 1, 0, fill=fill)
            pdf.cell(40, 10, f"{count}", 1, 0, "C", fill=fill)
            pdf.cell(50, 10, "Verified", 1, 1, "C", fill=fill)
            fill = not fill
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(190, 10, "CRITICAL: No Safety Equipment Detected", 1, 1, "C")
    
    file_name = f"ISRO_Report_{timestamp.replace(':', '').replace(' ', '_')}.pdf"
    pdf.output(file_name)
    return file_name

# ==========================================
# 🖥️ ENHANCED SIDEBAR
# ==========================================
with st.sidebar:
    # Header with logo
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=50)
    with col2:
        st.markdown("### ISRO COMMAND")
        st.markdown("**Safety Monitor v2.0**")
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 🟢 SYSTEM STATUS")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("AI Core", "Online", delta="Active")
    with status_col2:
        st.metric("Sensors", "Nominal", delta="Ready")
    
    st.markdown("---")
    
    # Configuration
    st.markdown("### ⚙️ MISSION CONFIG")
    
    # --- MODEL SWITCHER ---
    model_source_type = st.radio("Select Model Source:", ["Use Default (yolov8n)", "Upload Custom .pt"])
    
    if model_source_type == "Upload Custom .pt":
        uploaded_model = st.file_uploader("Upload .pt file", type=["pt"])
        if uploaded_model:
            with open("temp_custom_model.pt", "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = "temp_custom_model.pt"
            st.success("✅ Custom Brain Loaded!")
        else:
            model_path = None
    else:
        if os.path.exists("best.pt"):
            model_path = "best.pt"
            st.success(f"✅ Loaded Trained Model: {model_path}")
        else:
            model_path = "yolov8n.pt"
            st.info("ℹ️ Using Base Model (yolov8n)")

    confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.45)
    mode = st.radio("📡 INPUT MODE", ["Live Sentinel Feed", "Static Image Audit"])
    
    # Additional Controls
    with st.expander("🔧 Advanced Settings"):
        enable_voice = st.checkbox("Voice Alerts", value=True)
        alert_sensitivity = st.selectbox("Alert Level", ["Low", "Medium", "High"])
        auto_report = st.checkbox("Auto-generate Reports")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### 🚀 QUICK ACTIONS")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("🔄 Reboot", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    with action_col2:
        if st.button("📊 Report", use_container_width=True):
            st.session_state.generate_report = True

# ==========================================
# 🚀 ENHANCED MAIN DASHBOARD
# ==========================================
# Header Section
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
with header_col1:
    st.title("🛰️ ISRO SAFETY MONITOR")
    st.markdown("#### Bharatiya Antariksha Station • Gaganyaan Mission")
with header_col2:
    st.metric("Mission Time", datetime.datetime.now().strftime("%H:%M:%S"))
with header_col3:
    st.metric("Orbit", "LEO-400km", delta="Stable")

# Boot Sequence
if 'booted' not in st.session_state:
    with st.status("🚀 INITIATING ISRO SAFETY PROTOCOLS...", expanded=True) as status:
        st.write("🛰️ Establishing satellite uplink...")
        time.sleep(1)
        st.write("🧠 Loading neural network weights...")
        time.sleep(1)
        st.write("🔐 Authenticating mission credentials...")
        time.sleep(1)
        st.write("📡 Calibrating sensor arrays...")
        time.sleep(1)
        st.write("✅ All systems nominal")
        status.update(label="🚀 MISSION READY - Systems Online", state="complete", expanded=False)
    st.session_state['booted'] = True

# Load Model
if model_path:
    model = load_model(model_path)
else:
    st.stop()

# Initialize Session State
if 'mission_log' not in st.session_state:
    st.session_state['mission_log'] = []

# ==========================================
# 🎯 CUSTOM GAUGE COMPONENT (No Plotly)
# ==========================================
def create_safety_gauge(score):
    if score < 40:
        color = "#FF2E63"
        status = "CRITICAL"
    elif score < 70:
        color = "#FFBD69"
        status = "MODERATE"
    else:
        color = "#00FFC2"
        status = "EXCELLENT"
    
    gauge_html = f"""
    <div class="gauge-container">
        <div class="gauge-label">SAFETY SCORE</div>
        <div class="gauge-value" style="color: {color}; text-shadow: 0 0 10px {color};">
            {score}%
        </div>
        <div style="color: {color}; font-weight: bold; font-size: 1.2rem;">
            {status}
        </div>
        <div style="background: #2A2F45; height: 20px; border-radius: 10px; margin: 20px 0;">
            <div style="background: {color}; height: 100%; width: {score}%; border-radius: 10px; 
                      box-shadow: 0 0 10px {color}; transition: all 0.5s ease;">
            </div>
        </div>
    </div>
    """
    return gauge_html

# ==========================================
# 🎯 MODE 1: LIVE WEBCAM (ENHANCED)
# ==========================================
if mode == "Live Sentinel Feed":
    # Top Metrics Row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fps_metric = st.empty() # Placeholder
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        obj_metric = st.empty() # Placeholder
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        score_metric = st.empty() # Placeholder
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        alert_metric = st.empty() # Placeholder
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔴 LIVE SENTINEL FEED")
        run = st.toggle('🎯 ACTIVATE SENTINEL MODE', value=True)
        
        # ✅ FIX 2: STABLE CONTAINER FOR VIDEO
        FRAME_WINDOW = st.empty()
    
    with col2:
        # Status Panel
        st.markdown("### 📋 MISSION STATUS")
        status_container = st.empty()
        
        # Real-time Charts
        st.markdown("### 📈 TELEMETRY DATA")
        chart_tab1, chart_tab2 = st.tabs(["Confidence", "Objects"])
        
        with chart_tab1:
            conf_chart = st.empty()
        with chart_tab2:
            obj_chart = st.empty()
        
        # Equipment Status
        st.markdown("### 🛠️ EQUIPMENT STATUS")
        equip_container = st.empty()
        
        # Mission Log
        st.markdown("### 📝 MISSION LOG")
        log_container = st.empty()
    
    # Camera Processing
    if run:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        last_alert_time = 0
        alert_cooldown = 5
        confidence_data = []
        object_count_data = []
        safety_equipment = {}
        
        while run:
            ret, frame = camera.read()
            if not ret or frame is None:
                st.warning("🚨 Camera feed interrupted")
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # AI Inference
            start_time = time.time()
            results = model.predict(frame, conf=confidence, verbose=False)
            end_time = time.time()
            
            # Process results
            plotted_frame = results[0].plot()
            # ✅ UPDATE FRAME IN STABLE CONTAINER
            FRAME_WINDOW.image(plotted_frame, use_container_width=True)
            
            detections = results[0].boxes
            class_names = results[0].names
            
            # Update metrics
            current_fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            current_objects = len(detections) if detections is not None else 0
            
            # Update safety equipment tracking
            safety_equipment = {}
            if detections is not None:
                for box in detections:
                    class_id = int(box.cls.item())
                    class_name = class_names[class_id]
                    safety_equipment[class_name] = safety_equipment.get(class_name, 0) + 1
            
            # Calculate safety score
            safety_score = min(100, len(safety_equipment) * 25)
            
            # Update charts
            confidence_data.append(current_fps)
            object_count_data.append(current_objects)
            if len(confidence_data) > 50:
                confidence_data.pop(0)
                object_count_data.pop(0)
            
            # Update metrics in real-time (Using Placeholders)
            fps_metric.metric("🔄 System FPS", f"{current_fps:.1f}")
            obj_metric.metric("📊 Objects Tracked", str(current_objects))
            score_metric.metric("🛡️ Safety Score", f"{safety_score}%")
            
            # Update status
            if "fire" in str(safety_equipment.keys()).lower() and "fireextinguisher" not in safety_equipment:
                alert_metric.metric("⚠️ Alert Level", "CRITICAL", delta="Red", delta_color="inverse")
                status_container.markdown('<div class="warning-box">🚨 CRITICAL: Fire detected without extinguisher!</div>', unsafe_allow_html=True)
                if enable_voice and (time.time() - last_alert_time > alert_cooldown):
                    speak_warning("Critical alert! Fire detected without safety equipment!")
                    last_alert_time = time.time()
            else:
                alert_metric.metric("⚠️ Alert Level", "NOMINAL", delta="Green")
                status_container.markdown('<div class="success-box">✅ ALL SYSTEMS NOMINAL</div>', unsafe_allow_html=True)
            
            # Update equipment status
            equip_text = "\n".join([f"• {item}: {count} ✅" for item, count in safety_equipment.items()])
            if not equip_text:
                equip_text = "• No equipment detected"
            equip_container.text(equip_text)
            
            # Update mission log
            if safety_equipment:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] Detected: {', '.join(safety_equipment.keys())}"
                if not st.session_state['mission_log'] or st.session_state['mission_log'][-1] != log_entry:
                    st.session_state['mission_log'].append(log_entry)
            
            # Display last 5 log entries
            log_display = "\n".join(st.session_state['mission_log'][-5:][::-1])
            log_container.code(log_display, language="text")
            
            # Update charts
            conf_chart.line_chart(confidence_data)
            obj_chart.line_chart(object_count_data)
        
        camera.release()

# ==========================================
# 🎯 MODE 2: STATIC IMAGE AUDIT (ENHANCED)
# ==========================================
else:
    st.markdown("### 📸 STATIC IMAGE SAFETY AUDIT")
    
    upload_col1, upload_col2 = st.columns([1, 2])
    
    with upload_col1:
        st.markdown("#### Upload Inspection Image")
        uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
        
        if uploaded_file:
            st.success("✅ Image uploaded successfully")
            
            # Analysis options
            st.markdown("#### 🔍 Analysis Options")
            detailed_analysis = st.checkbox("Detailed Analysis", value=True)
            generate_heatmap = st.checkbox("Generate Heatmap", value=False)
            
            if st.button("🚀 Start Safety Audit", use_container_width=True):
                st.session_state.analyze_image = True
    
    with upload_col2:
        if uploaded_file and hasattr(st.session_state, 'analyze_image'):
            image = Image.open(uploaded_file)
            
            # Create analysis tabs
            tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "🖼️ Processed Image", "📈 Safety Metrics"])
            
            with tab1:
                results = model.predict(image, conf=confidence)
                plotted = results[0].plot()
                
                detections = results[0].boxes
                class_names = results[0].names
                
                safety_equipment = {}
                if detections is not None:
                    for box in detections:
                        class_id = int(box.cls.item())
                        class_name = class_names[class_id]
                        safety_equipment[class_name] = safety_equipment.get(class_name, 0) + 1
                
                safety_score = min(100, len(safety_equipment) * 25)
                
                # Safety Assessment
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🛡️ Safety Assessment")
                    st.metric("Overall Safety Score", f"{safety_score}%")
                    
                    if safety_score >= 80:
                        st.success("✅ EXCELLENT: All safety standards met")
                    elif safety_score >= 60:
                        st.warning("⚠️ GOOD: Minor improvements needed")
                    else:
                        st.error("🚨 POOR: Critical safety issues detected")
                
                with col2:
                    st.markdown("#### 📋 Detected Equipment")
                    for item, count in safety_equipment.items():
                        st.write(f"• {item}: {count}")
                    
                    if not safety_equipment:
                        st.error("No safety equipment detected!")
            
            with tab2:
                # Check if user wants the heatmap
                if generate_heatmap:
                    st.info("🔥 Generating density heatmap based on detection clusters...")
                    # ✅ HEATMAP FIX: Call function correctly
                    heatmap_result = generate_heatmap_overlay(image, results[0].boxes)
                    st.image(heatmap_result, caption="AI Density Heatmap", use_container_width=True)
                else:
                    # Show standard bounding boxes
                    st.image(plotted, caption="AI Analysis Result", use_container_width=True)
            
            with tab3:
                # Custom safety gauge
                st.markdown(create_safety_gauge(safety_score), unsafe_allow_html=True)
                
                # Safety thresholds
                st.markdown("#### 🎯 Safety Thresholds")
                st.markdown("""
                - 🟢 **70-100%**: EXCELLENT - All safety equipment present
                - 🟡 **40-69%**: MODERATE - Some equipment missing
                - 🔴 **0-39%**: CRITICAL - Major safety concerns
                """)
            
            # Report Generation
            st.markdown("---")
            report_col1, report_col2 = st.columns([3, 1])
            
            with report_col2:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating professional report..."):
                        pdf_file = generate_pdf(safety_score, safety_equipment, timestamp, st.session_state['mission_log'])
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                "⬇️ Download Safety Report", 
                                f, 
                                file_name=pdf_file, 
                                mime="application/pdf",
                                use_container_width=True
                            )

# ==========================================
# 🎯 FOOTER
# ==========================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**ISRO - Indian Space Research Organisation**")
with footer_col2:
    st.markdown("Bharatiya Antariksha Station Program")
with footer_col3:
    st.markdown(f"System Version 2.0 • {datetime.datetime.now().year}")