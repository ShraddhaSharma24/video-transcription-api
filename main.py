import whisper
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import timedelta
from PIL import Image
import uvicorn

class VideoTranscriptionAgent:
    def __init__(self, model_size="tiny"):
        """
        Initialize the transcription agent
        model_size options: tiny, base, small, medium, large
        'tiny' is fastest (~1GB RAM, ~32x faster than real-time)
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")

    def extract_audio(self, video_path, audio_output_path="temp_audio.wav"):
        """Extract audio from video file"""
        print(f"Extracting audio from {video_path}...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path, logger=None)
        video.close()
        print(f"Audio extracted to {audio_output_path}")
        return audio_output_path

    def extract_frame(self, video_path, timestamp, output_path):
        """Extract a single frame from video at specific timestamp"""
        try:
            video = VideoFileClip(video_path)
            frame = video.get_frame(timestamp)
            video.close()

            # Save frame as image
            img = Image.fromarray(frame)
            img.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error extracting frame at {timestamp}s: {e}")
            return None



    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS.mmm format"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def extract_audio_chunk(self, audio_path, start_time, end_time, output_path):
        """Extract a specific time range from audio file"""
        audio = AudioSegment.from_wav(audio_path)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        chunk = audio[start_ms:end_ms]
        chunk.export(output_path, format="wav")
        return output_path

    def transcribe_video(self, video_path, language=None, output_dir="audio_chunks",
                        frames_dir="video_frames", frame_position="middle"):
        """
        Transcribe video with timestamps, extract audio chunks and frames
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        # Extract audio
        audio_path = self.extract_audio(video_path)

        # Transcribe with timestamps
        print("Transcribing audio...")
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )

        # Format results
        formatted_result = {
            "full_text": result["text"],
            "language": result["language"],
            "segments": []
        }

        print("Extracting audio chunks and video frames...")
        for segment in result["segments"]:
            segment_id = segment["id"]

            chunk_filename = f"segment_{segment_id:04d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)

            frame_filename = f"frame_{segment_id:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            # Calculate frame timestamp
            if frame_position == "start":
                frame_timestamp = segment["start"]
            elif frame_position == "end":
                frame_timestamp = segment["end"]
            else:
                frame_timestamp = (segment["start"] + segment["end"]) / 2

            # Extract audio chunk and frame
            self.extract_audio_chunk(audio_path, segment["start"], segment["end"], chunk_path)
            frame_extracted = self.extract_frame(video_path, frame_timestamp, frame_path)

            formatted_segment = {
                "id": segment["id"],
                "start": self.format_timestamp(segment["start"]),
                "end": self.format_timestamp(segment["end"]),
                "start_seconds": segment["start"],
                "end_seconds": segment["end"],
                "duration_seconds": segment["end"] - segment["start"],
                "text": segment["text"].strip(),
                "audio_file": chunk_filename,
                "audio_path": chunk_path,
                "frame_file": frame_filename,
                "frame_path": frame_path if frame_extracted else None,
                "frame_timestamp": self.format_timestamp(frame_timestamp),
                "frame_timestamp_seconds": frame_timestamp
            }

            formatted_result["segments"].append(formatted_segment)

        # Clean up temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        print(f"Created {len(formatted_result['segments'])} segments")
        return formatted_result

# Initialize FastAPI app
app = FastAPI(
    title="Video Transcription API",
    version="3.0.0",
    description="API for video transcription with audio chunks and frame extraction"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent on startup
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    agent = VideoTranscriptionAgent(model_size="tiny")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Transcription API with Audio Chunks & Frame Extraction",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "audio": "/audio/{filename} (GET)",
            "frame": "/frame/{filename} (GET)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    import psutil
    import platform
    
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "model": "whisper-tiny",
        "agent_loaded": agent is not None,
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_mb": round(memory.total / (1024 * 1024), 2),
            "memory_available_mb": round(memory.available / (1024 * 1024), 2),
            "memory_percent": memory.percent
        }
    }

@app.post("/transcribe")
async def transcribe(
    video: UploadFile = File(...),
    language: str = None,
    frame_position: str = "middle"
):
    """
    Transcribe video and extract audio chunks, frames, and video clips
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    temp_video_path = None
    
    try:
        # Validate file size (max 50MB to prevent timeouts)
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        if file_size_mb > 50:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.2f}MB. Maximum size is 50MB."
            )
        
        print(f"Processing video: {video.filename} ({file_size_mb:.2f}MB)")
        
        # Save uploaded video
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(content)

        print("Starting transcription...")
        
        # Process video
        result = agent.transcribe_video(
            temp_video_path,
            language=language,
            output_dir="audio_chunks",
            frames_dir="video_frames",
            frame_position=frame_position
        )
        
        print(f"Transcription complete: {len(result['segments'])} segments")

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Always clean up temp video
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                print(f"Cleaned up {temp_video_path}")
            except Exception as e:
                print(f"Failed to clean up {temp_video_path}: {e}")

@app.get("/audio/{filename}")
async def get_audio_chunk(filename: str):
    """Download a specific audio chunk"""
    audio_path = os.path.join("audio_chunks", filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/wav", filename=filename)

@app.get("/frame/{filename}")
async def get_frame(filename: str):
    """Download a specific video frame"""
    frame_path = os.path.join("video_frames", filename)
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame file not found")
    return FileResponse(frame_path, media_type="image/jpeg", filename=filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


