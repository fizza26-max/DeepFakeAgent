# app.py
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64
import time
import os

st.set_page_config(page_title="DeepFakeAgent - Frontend (Streamlit)", layout="wide")

# --- Helper functions -------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

OUTPUT_DIR = ensure_dir(Path("recordings"))
GALLERY_DIR = Path("images")  # you can populate this with sample thumbnails

def save_base64_video(b64_string: str, out_path: Path):
    # b64_string may include the data prefix "data:video/webm;base64,..."
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    video_bytes = base64.b64decode(b64_string)
    out_path.write_bytes(video_bytes)
    return out_path

def fake_backend_process(input_video_path: Path, selected_image_path: Path = None) -> Path:
    """
    Placeholder: replace this with the real processing that your notebook does.
    For demo, this function just copies the recorded file to a "generated" file
    and returns its path after a small delay to simulate processing time.
    """
    generated_dir = ensure_dir(Path("generated"))
    out = generated_dir / f"generated_{int(time.time())}.webm"
    # simple copy for placeholder
    out.write_bytes(input_video_path.read_bytes())
    time.sleep(0.5)
    return out

# --- UI --------------------------------------------------------------------
st.title("🎥 DeepFakeAgent — Streamlit Frontend (Prototype)")
st.write("Select an image (or upload your own), then record a video (camera + mic). The recorded video will be saved and processed.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1) Image gallery & upload")
    # show gallery if available
    if GALLERY_DIR.exists():
        thumbs = sorted([p for p in GALLERY_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif"]])
        if thumbs:
            # show thumbnails as selectable radio buttons with small preview
            options = ["(none)"] + [p.name for p in thumbs]
            choice = st.selectbox("Pick one of the gallery images", options)
            selected_image = None
            if choice != "(none)":
                selected_image = GALLERY_DIR / choice
                st.image(str(selected_image), caption=choice, use_column_width=True)
        else:
            st.info("No images in `images/` folder. Upload below or create an `images/` folder.")
            selected_image = None
    else:
        st.info("No `images/` folder found. You can upload an image below.")
        selected_image = None

    st.write("---")
    uploaded = st.file_uploader("Or upload an image to use (optional)", type=["png", "jpg", "jpeg", "gif"])
    if uploaded is not None:
        upload_path = ensure_dir(Path("uploads")) / f"{int(time.time())}_{uploaded.name}"
        upload_path.write_bytes(uploaded.getbuffer())
        st.success("Uploaded image saved.")
        st.image(str(upload_path), caption=uploaded.name, use_column_width=True)
        selected_image = upload_path

with col2:
    st.header("2) Record video (camera + microphone)")
    st.markdown(
        """
- Press **Start recording**. Allow camera & microphone access in the browser.
- Press **Stop** when finished. The recorded file will be uploaded to the Streamlit app and saved as a `.webm` file.
"""
    )

    RECORD_BUTTON = st.empty()
    REC_STATUS = st.empty()

    # Embedded HTML + JS recorder
    # The HTML will post a message back to Streamlit with `{ "video": "<base64-data>" }`
    recorder_html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
    </head>
    <body>
      <div style="font-family: sans-serif;">
        <video id="preview" autoplay muted playsinline style="width:100%; max-height:360px; background:#000;"></video>
        <div style="margin-top:8px;">
          <button id="startBtn">Start recording</button>
          <button id="stopBtn" disabled>Stop</button>
          <span id="status" style="margin-left:12px;"></span>
        </div>
      </div>

      <script>
        const startBtn = document.getElementById("startBtn");
        const stopBtn = document.getElementById("stopBtn");
        const statusSpan = document.getElementById("status");
        const preview = document.getElementById("preview");

        let mediaRecorder;
        let recordedChunks = [];

        async function startRecording() {
          recordedChunks = [];
          statusSpan.innerText = "Requesting camera/mic permission...";
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            window.localStream = stream;
            preview.srcObject = stream;
            preview.play();
            statusSpan.innerText = "Recording...";
            startBtn.disabled = true;
            stopBtn.disabled = false;

            const options = { mimeType: 'video/webm; codecs=vp8,opus' };
            mediaRecorder = new MediaRecorder(stream, options);

            mediaRecorder.ondataavailable = function(e) {
              if (e.data && e.data.size > 0) {
                recordedChunks.push(e.data);
              }
            };

            mediaRecorder.onstop = function() {
              const blob = new Blob(recordedChunks, { type: 'video/webm' });
              const reader = new FileReader();
              reader.onloadend = function() {
                const base64data = reader.result; // includes data:...;base64,...
                // Post message to parent (Streamlit will capture this and return it)
                const payload = { 'video': base64data };
                window.parent.postMessage(payload, "*");
              };
              reader.readAsDataURL(blob);
            };

            mediaRecorder.start();
          } catch (err) {
            console.error(err);
            statusSpan.innerText = "Error: " + err.message;
          }
        }

        function stopRecording() {
          statusSpan.innerText = "Stopping...";
          stopBtn.disabled = true;
          startBtn.disabled = false;
          if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
          }
          if (window.localStream) {
            window.localStream.getTracks().forEach(t => t.stop());
          }
          statusSpan.innerText = "Stopped. Uploading to app...";
        }

        startBtn.addEventListener("click", startRecording);
        stopBtn.addEventListener("click", stopRecording);

        // For mobile orientation handling
        window.addEventListener("orientationchange", function(){ setTimeout(()=>preview.play(), 300); });
      </script>
    </body>
    </html>
    """

    # components.html returns the data object posted by the embedded page when it calls postMessage()
    result = components.html(recorder_html, height=480)

    # result will be a dict-like object containing the posted payload if recorder posted it.
    if result is not None:
        # The result from components.html may show up as a dict or string.
        # We try to extract a base64 video field robustly.
        video_b64 = None
        if isinstance(result, dict):
            video_b64 = result.get("video") or result.get("base64") or result.get("data")
        elif isinstance(result, str):
            # attempt to see if it's JSON-ish
            try:
                import json
                parsed = json.loads(result)
                video_b64 = parsed.get("video") or parsed.get("data")
            except Exception:
                # fallback: take the whole string if it looks like data:video
                if result.startswith("data:video"):
                    video_b64 = result

        if video_b64:
            try:
                timestamp = int(time.time())
                out_path = OUTPUT_DIR / f"recording_{timestamp}.webm"
                save_base64_video(video_b64, out_path)
                st.success(f"Saved recording to `{out_path}`")
                st.video(str(out_path))
                REC_STATUS.success("Recording saved and shown above.")
                # Run backend processing (placeholder)
                with st.spinner("Running backend processing..."):
                    generated = fake_backend_process(out_path, selected_image)
                st.success("Processing complete — showing generated video")
                st.video(str(generated))
            except Exception as e:
                st.error(f"Failed to save/process video: {e}")
        else:
            st.warning("No valid video data returned from recorder. Try again or use another browser.")
