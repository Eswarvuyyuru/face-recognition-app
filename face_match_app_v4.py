import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import pandas as pd
import logging
from pathlib import Path
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from PIL import Image
import threading
import queue

# Set page config FIRST
st.set_page_config(
    page_title="🔍 Face Match System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced model loading with error handling
@st.cache_resource
def load_model():
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app, True
    except Exception as e:
        st.error(f"Failed to load face analysis model: {str(e)}")
        return None, False

app, model_loaded = load_model()

# Enhanced UI
st.title("🎯 Advanced Face Recognition & Alert System")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    similarity_threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.2, 
        max_value=0.8, 
        value=0.4, 
        step=0.01,
        help="Higher values = stricter matching"
    )
    
    detection_confidence = st.slider(
        "Face Detection Confidence",
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    frame_skip = st.number_input(
        "Process Every N Frames (Video)",
        min_value=1,
        max_value=30,
        value=5,
        help="Skip frames to improve performance"
    )
    
    max_faces_per_frame = st.number_input(
        "Max Faces Per Frame",
        min_value=1,
        max_value=20,
        value=10
    )

if not model_loaded:
    st.stop()

# Enhanced session state management
def init_session_state():
    defaults = {
        "results_log": [],
        "video_processed": False,
        "processed_video_path": "",
        "reference_embeddings": [],
        "total_matches": 0
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Enhanced reference face processing
def process_reference_images(ref_images):
    """Process and validate reference images"""
    embeddings = []
    processed_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ref_file in enumerate(ref_images):
        status_text.text(f"Processing {ref_file.name}...")
        
        try:
            ref_bytes = ref_file.read()
            ref_img = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if ref_img is None:
                st.warning(f"⚠️ Could not read image: {ref_file.name}")
                continue
                
            faces = app.get(ref_img, max_num=1)  # Get only the largest face
            
            if faces:
                face = faces[0]  # Take the first (largest) face
                # Validate face quality
                if face.det_score < detection_confidence:
                    st.warning(f"⚠️ Low quality face detected in {ref_file.name}")
                    continue
                    
                embedding = face.embedding / np.linalg.norm(face.embedding)
                embeddings.append({
                    'name': ref_file.name,
                    'embedding': embedding,
                    'quality_score': face.det_score
                })
                processed_count += 1
            else:
                st.warning(f"⚠️ No face detected in {ref_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {ref_file.name}: {str(e)}")
            
        progress_bar.progress((i + 1) / len(ref_images))
    
    progress_bar.empty()
    status_text.empty()
    
    return embeddings, processed_count

# Enhanced match logging
def log_match(source, frame_num, timestamp, query_name, similarity, face_crop, full_frame, face_bbox):
    """Enhanced match logging with better image handling"""
    try:
        # Create unique filename
        img_name = f"match_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        img_path = os.path.join(tempfile.gettempdir(), img_name)
        
        # Save full frame with bounding box
        annotated_frame = full_frame.copy()
        cv2.rectangle(annotated_frame, (face_bbox[0], face_bbox[1]), 
                     (face_bbox[2], face_bbox[3]), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{query_name}: {similarity:.3f}", 
                   (face_bbox[0], face_bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(img_path, annotated_frame)
        
        # Log the match
        match_data = {
            'source': source,
            'frame_num': frame_num,
            'timestamp': timestamp,
            'person': query_name,
            'similarity': similarity,
            'image_path': img_path,
            'bbox': face_bbox.tolist()
        }
        
        st.session_state.results_log.append(match_data)
        st.session_state.total_matches += 1
        
        # Show alert
        st.success(f"🎯 **MATCH FOUND!** {query_name} at {timestamp} (Similarity: {similarity:.3f})")
        
    except Exception as e:
        logger.error(f"Error logging match: {str(e)}")

# Cache uploaded video with better handling
@st.cache_data
def save_uploaded_video(uploaded_video):
    """Save uploaded video with proper cleanup"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(uploaded_video.read())
            return tmp_vid.name
    except Exception as e:
        st.error(f"Error saving video: {str(e)}")
        return None

# Main application
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Reference Images")
    ref_images = st.file_uploader(
        "Upload Reference Face Images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Upload clear, front-facing photos"
    )

with col2:
    st.header("🎮 Mode Selection")
    mode = st.radio(
        "Choose Processing Mode", 
        ["Live Webcam", "Video Upload"],
        help="Select whether to process live camera feed or uploaded video"
    )

# Process reference images
if ref_images:
    with st.expander("🔍 Reference Image Processing", expanded=True):
        embeddings, processed_count = process_reference_images(ref_images)
        st.session_state.reference_embeddings = embeddings
        
        if processed_count > 0:
            st.success(f"✅ Successfully processed {processed_count}/{len(ref_images)} reference images")
            
            # Show reference image previews
            cols = st.columns(min(processed_count, 4))
            for i, emb_data in enumerate(embeddings[:4]):
                with cols[i % 4]:
                    st.text(f"{emb_data['name']}")
                    st.text(f"Quality: {emb_data['quality_score']:.2f}")
        else:
            st.error("❌ No valid reference faces found!")
            st.stop()

    # PROCESSING MODES
    st.markdown("---")
    
    if mode == "Live Webcam":
        st.header("📹 Live Camera Feed")
        
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.frame_count = 0
                self.fps = 30
                self.last_alert_time = {}  # Prevent spam alerts
                
            def transform(self, frame):
                self.frame_count += 1
                img = frame.to_ndarray(format="bgr24")
                
                try:
                    faces = app.get(img, max_num=max_faces_per_frame)
                    current_time = datetime.datetime.now()
                    
                    for face in faces:
                        if face.det_score < detection_confidence:
                            continue
                            
                        box = face.bbox.astype(int)
                        emb = face.embedding / np.linalg.norm(face.embedding)
                        
                        for emb_data in st.session_state.reference_embeddings:
                            similarity = np.dot(emb, emb_data['embedding'])
                            
                            if similarity > similarity_threshold:
                                # Prevent spam alerts (max 1 per person per 5 seconds)
                                alert_key = f"{emb_data['name']}_{box[0]}_{box[1]}"
                                last_alert = self.last_alert_time.get(alert_key, 0)
                                
                                if (current_time.timestamp() - last_alert) > 5:
                                    timestamp = str(datetime.timedelta(seconds=self.frame_count / self.fps))
                                    face_crop = img[box[1]:box[3], box[0]:box[2]]
                                    
                                    log_match("Live Camera", self.frame_count, timestamp, 
                                            emb_data['name'], similarity, face_crop, img.copy(), box)
                                    
                                    self.last_alert_time[alert_key] = current_time.timestamp()
                                
                                # Draw bounding box
                                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                                cv2.putText(img, f"{emb_data['name']}: {similarity:.2f}", 
                                           (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                break
                                
                except Exception as e:
                    logger.error(f"Error in video transform: {str(e)}")
                
                return img

        # WebRTC streamer with better configuration
        webrtc_ctx = webrtc_streamer(
            key="face_detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    else:  # Video Upload Mode
        st.header("🎬 Video Processing")
        
        uploaded_video = st.file_uploader(
            "Upload Video File", 
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload your surveillance video"
        )
        
        if uploaded_video:
            video_path = save_uploaded_video(uploaded_video)
            
            if video_path and not st.session_state.video_processed:
                if st.button("🚀 Start Processing", type="primary"):
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Create output video
                        out_path = os.path.join(tempfile.gettempdir(), 
                                              f"processed_{uploaded_video.name}")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_writer = cv2.VideoWriter(out_path, fourcc, fps, 
                                                   (frame_width, frame_height))
                        
                        # Processing UI
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        matches_text = st.empty()
                        
                        frame_num = 0
                        matches_found = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Skip frames for performance
                            if frame_num % frame_skip != 0:
                                out_writer.write(frame)
                                frame_num += 1
                                continue
                            
                            status_text.text(f"Processing frame {frame_num}/{total_frames}")
                            
                            try:
                                faces = app.get(frame, max_num=max_faces_per_frame)
                                
                                for face in faces:
                                    if face.det_score < detection_confidence:
                                        continue
                                        
                                    box = face.bbox.astype(int)
                                    emb = face.embedding / np.linalg.norm(face.embedding)
                                    
                                    for emb_data in st.session_state.reference_embeddings:
                                        similarity = np.dot(emb, emb_data['embedding'])
                                        
                                        if similarity > similarity_threshold:
                                            timestamp = str(datetime.timedelta(seconds=frame_num / fps))
                                            face_crop = frame[box[1]:box[3], box[0]:box[2]]
                                            
                                            log_match(uploaded_video.name, frame_num, timestamp,
                                                    emb_data['name'], similarity, face_crop, 
                                                    frame.copy(), box)
                                            
                                            matches_found += 1
                                            matches_text.text(f"Matches found: {matches_found}")
                                            
                                            # Annotate frame
                                            cv2.rectangle(frame, (box[0], box[1]), 
                                                        (box[2], box[3]), (0, 255, 0), 2)
                                            cv2.putText(frame, f"{emb_data['name']}: {similarity:.2f}",
                                                       (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                       0.6, (0, 255, 0), 2)
                                            break
                                            
                            except Exception as e:
                                logger.error(f"Error processing frame {frame_num}: {str(e)}")
                            
                            out_writer.write(frame)
                            frame_num += 1
                            
                            # Update progress
                            progress = min(frame_num / total_frames, 1.0)
                            progress_bar.progress(progress)
                        
                        cap.release()
                        out_writer.release()
                        
                        st.session_state.video_processed = True
                        st.session_state.processed_video_path = out_path
                        
                        progress_bar.empty()
                        status_text.empty()
                        matches_text.empty()
                        
                        st.success(f"✅ Video processing complete! Found {matches_found} matches.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
            
            # Download processed video
            if st.session_state.video_processed and os.path.exists(st.session_state.processed_video_path):
                with open(st.session_state.processed_video_path, "rb") as vfile:
                    st.download_button(
                        "🎥 Download Processed Video",
                        vfile.read(),
                        file_name=f"processed_{uploaded_video.name}",
                        mime="video/mp4"
                    )

    # RESULTS SECTION
    st.markdown("---")
    st.header("📊 Results Dashboard")
    
    if st.session_state.results_log:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(st.session_state.results_log))
        
        with col2:
            unique_persons = len(set(match['person'] for match in st.session_state.results_log))
            st.metric("Unique Persons", unique_persons)
        
        with col3:
            avg_similarity = np.mean([match['similarity'] for match in st.session_state.results_log])
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
        
        with col4:
            max_similarity = max(match['similarity'] for match in st.session_state.results_log)
            st.metric("Best Match", f"{max_similarity:.3f}")
        
        # Results table
        df_data = []
        for match in st.session_state.results_log:
            df_data.append({
                'Source': match['source'],
                'Frame': match['frame_num'],
                'Timestamp': match['timestamp'],
                'Person': match['person'],
                'Similarity': round(match['similarity'], 3),
                'Image Path': match['image_path']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Export functionality
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel export
            xlsx_path = os.path.join(tempfile.gettempdir(), "face_matches.xlsx")
            df.to_excel(xlsx_path, index=False)
            
            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "📥 Download Excel Report",
                    f.read(),
                    file_name="face_matches.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV Report",
                csv,
                file_name="face_matches.csv",
                mime="text/csv"
            )
        
        # Image gallery
        st.subheader("🖼️ Match Gallery")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_person = st.selectbox(
                "Filter by Person",
                ["All"] + list(set(match['person'] for match in st.session_state.results_log))
            )
        
        with col2:
            min_similarity = st.slider(
                "Minimum Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        
        # Display filtered matches
        filtered_matches = st.session_state.results_log
        if selected_person != "All":
            filtered_matches = [m for m in filtered_matches if m['person'] == selected_person]
        filtered_matches = [m for m in filtered_matches if m['similarity'] >= min_similarity]
        
        # Display in grid
        cols_per_row = 3
        for i in range(0, len(filtered_matches), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, match in enumerate(filtered_matches[i:i+cols_per_row]):
                with cols[j]:
                    try:
                        if os.path.exists(match['image_path']):
                            img = cv2.imread(match['image_path'])
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                st.image(
                                    img_rgb,
                                    caption=f"{match['person']} | {match['timestamp']} | Sim: {match['similarity']:.3f}",
                                    use_container_width=True
                                )
                            else:
                                st.error("Could not load image")
                        else:
                            st.warning("Image file not found")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
    else:
        st.info("🔍 No matches detected yet. Upload reference images and start processing!")

    # Control buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All Data", type="secondary"):
            for key in ["results_log", "video_processed", "processed_video_path", 
                       "reference_embeddings", "total_matches"]:
                if key in st.session_state:
                    st.session_state[key] = [] if key in ["results_log", "reference_embeddings"] else (0 if key == "total_matches" else False if key == "video_processed" else "")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Matches", type="secondary"):
            st.session_state.results_log = []
            st.session_state.total_matches = 0
            st.rerun()
    
    with col3:
        if st.button("💾 Save Session", type="secondary"):
            # Could implement session saving functionality here
            st.info("Session save functionality can be implemented based on requirements")

else:
    st.info("👤 Please upload at least one reference face image to get started.")
    st.markdown("""
    ### 📋 Quick Start Guide:
    1. **Upload Reference Images**: Clear, front-facing photos of people to identify
    2. **Configure Settings**: Adjust similarity threshold and detection confidence in sidebar  
    3. **Choose Mode**: Live camera feed or upload a video file
    4. **Start Detection**: System will automatically detect and alert on matches
    5. **Review Results**: Download reports and view matched screenshots
    """)

# Footer
st.markdown("---")
st.markdown("*Advanced Face Recognition System - Built with Streamlit & InsightFace*")