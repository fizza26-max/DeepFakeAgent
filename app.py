import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter
import io
import tempfile
import subprocess
import sys

st.set_page_config(page_title="Safe Avatar Animator UI", layout="wide")

st.title("Safe Avatar Animator — Streamlit UI (Consent-first, non-impersonating)")
st.markdown(
    """
    **Important:** I cannot provide tools that create real-time deepfakes of real people or clone voices without clear consent.
    Instead this UI focuses on *safe, ethical alternatives*: anonymization, stylized filters, and animating **user-supplied** fictional avatars.

    This Streamlit app is a **frontend**: model inference (animation, advanced TTS) should run in a backend service you control. The UI contains *placeholders* where you would call those services.
    """
)

# Sidebar: configuration
with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Mode", ["Live webcam (streamlit camera)", "Upload image"]) 
    show_debug = st.checkbox("Show debug / intermediate frames", value=False)
    actions = st.multiselect("Available actions (select one or more)",
                             ["Animate avatar (requires backend)", "Stylize (cartoonify)", "Anonymize (pixelate/blur faces)", "Text-to-speech from text"],
                             default=["Stylize (cartoonify)"])
    st.markdown("---")
    st.write("**Notes**")
    st.write("• This front-end **does not** include any face-swapping or voice cloning code.\n• Use only with images/avatars you own or explicit consent.")

col1, col2 = st.columns([1, 1])

# Capture or upload
if mode == "Live webcam (streamlit camera)":
    raw_image = st.camera_input("Point your camera at a neutral background")
else:
    raw_image = st.file_uploader("Upload a photo or avatar (PNG/JPEG)", type=["png", "jpg", "jpeg"])

# Avatar image (optional) - for animating a separate avatar image
avatar_file = st.file_uploader("(Optional) Upload an avatar image to animate (PNG/JPEG)", type=["png", "jpg", "jpeg"] , key="avatar")

# Text-to-speech input
if "Text-to-speech from text" in actions:
    tts_text = st.text_area("Text to synthesize (TTS)", value="Hello — this is a clearly synthetic voice.")
    tts_play = st.button("Synthesize & Play (local TTS)")

# Helper functions

@st.cache_data
def load_face_detector():
    # Use OpenCV's DNN face detector or Mediapipe in a real setup. Here we fallback to Haar cascades (fast, simple).
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cascade

face_cascade = load_face_detector()


def pil_to_cv2(image: Image.Image):
    arr = np.array(image.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def cartoonify(cv_img):
    # Simple stylization: bilateral filter + edge mask
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    num_down = 2  # downsample steps
    num_bilateral = 7  # bilateral filtering steps
    img_color = img_rgb.copy()
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    edges = cv2.adaptiveThreshold(img_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(img_color, edges)
    cartoon_bgr = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
    return cartoon_bgr


def anonymize_faces(cv_img, method='pixelate', blocks=15):
    img_out = cv_img.copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        roi = img_out[y:y+h, x:x+w]
        if method == 'pixelate':
            # downscale then upscale
            h0, w0 = roi.shape[:2]
            temp = cv2.resize(roi, (max(1, w0//blocks), max(1, h0//blocks)), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (w0, h0), interpolation=cv2.INTER_NEAREST)
            img_out[y:y+h, x:x+w] = pixelated
        else:
            # blur
            k = max(15, (w//3)|1)
            img_out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)
    return img_out


# PROCESS IMAGE
if raw_image is not None:
    bytes_data = raw_image.read()
    input_pil = Image.open(io.BytesIO(bytes_data)).convert('RGB')
    cv_img = pil_to_cv2(input_pil)

    results = []

    if "Stylize (cartoonify)" in actions:
        cartoon = cartoonify(cv_img)
        results.append(("Stylized (cartoon)", cartoon))

    if "Anonymize (pixelate/blur faces)" in actions:
        anon = anonymize_faces(cv_img, method='pixelate')
        results.append(("Anonymized (pixelate)", anon))

    if "Animate avatar (requires backend)" in actions:
        st.warning("Avatar animation requires a backend model. This UI will POST the camera frame + avatar image to your backend endpoint.")
        if avatar_file is None:
            st.info("Upload an avatar image to enable animation (this should be a *fictional* character or an image you own).")
        else:
            # show preview of target avatar
            avatar_pil = Image.open(avatar_file).convert('RGBA')
            st.image(avatar_pil, caption='Avatar (target)')
            if st.button("Request animation (POST to /animate)"):
                st.info("Preparing request payload and sending to backend... (this is a UI placeholder)")
                # prepare payload bytes
                with st.spinner("Assembling data..."):
                    avatar_bytes = avatar_file.read()
                    frame_bytes = bytes_data
                    # In a real app: send `frame_bytes` and `avatar_bytes` to your model server.
                    st.success("Payload prepared. Send these bytes to your animation service.\nExample: POST /animate with multipart/form-data fields 'frame' and 'avatar'.")
                    st.code(
                        """
                        # Example (python requests)
                        # files = {'frame': ('frame.jpg', frame_bytes, 'image/jpeg'),
                        #          'avatar': ('avatar.png', avatar_bytes, 'image/png')}
                        # r = requests.post('http://your-backend/animate', files=files)
                        """
                    )

    # show outputs
    st.subheader("Output preview")
    cols = st.columns(len(results) + 1)
    # Original
    cols[0].image(cv2_to_pil(cv_img), caption='Original')
    for i, (title, arr) in enumerate(results, start=1):
        cols[i].image(cv2_to_pil(arr), caption=title)

else:
    st.info("Waiting for camera input or upload...")

# Text-to-speech (local)
if "Text-to-speech from text" in actions and 'tts_text' in locals():
    if tts_play:
        st.info("Running local TTS (pyttsx3) — this produces a clearly synthetic voice. Do not use to impersonate real people.")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(tts_text)
            engine.runAndWait()
            st.success("Spoken locally on the server where Streamlit runs (note: many hosting services disable audio playback).")
        except Exception as e:
            st.error(f"Local TTS failed: {e}")
            st.write("As an alternative, export text to a file and use a local TTS tool on your workstation.")

# Debug / developer hints
with st.expander("Developer notes & next steps (for building backend)"):
    st.markdown(
        """
        **Where to attach models / servers**

        1. **Avatar animation**: run a dedicated model server that implements an endpoint such as `POST /animate` which accepts multipart form data: `frame` (caller camera frame) and `avatar` (target image). The model should only animate fictional avatars or images with explicit consent.\n
        2. **Advanced TTS**: If you plan to use open-source TTS (Coqui TTS, Mozilla TTS), do so with clearly synthetic voices and user consent.\n
        3. **Safety**: Log consent, display watermarks on outputs, and store audit trails.\n
        **This UI intentionally avoids providing any deepfake model weights or voice cloning code.**
        """
    )

st.markdown("---")
st.write("If you'd like, I can: \n• generate a ready-to-run Flask backend skeleton that exposes `/animate` and `/tts` endpoints (it will be *placeholders* — no cloning), or \n• adapt this UI to a React page instead of Streamlit.")

# Footer: legal & ethical reminder
st.caption("Use this tool only for lawful, ethical purposes. Do not use to impersonate, defraud, or deceive others.")
