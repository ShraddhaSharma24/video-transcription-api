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

    def extract_video_clip(self, video_path, start_time, end_time, output_path):
        """Extract a video segment WITHOUT audio"""
        try:
            video = VideoFileClip(video_path)
            clip = video.subclip(start_time, end_time)
            clip_no_audio = clip.without_audio()
            clip_no_audio.write_videofile(
                output_path,
                codec='libx264',
                audio=False,
                logger=None,
                preset='ultrafast'
            )
            video.close()
            clip.close()
            clip_no_audio.close()
            return output_path
        except Exception as e:
            print(f"Error extracting video clip from {start_time}s to {end_time}s: {e}")
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
                        frames_dir="video_frames", clips_dir="video_clips",
                        frame_position="middle"):
        """
        Transcribe video with timestamps, extract audio chunks, frames, and video clips
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)

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

        print("Extracting audio chunks, video frames, and video clips...")
        for segment in result["segments"]:
            segment_id = segment["id"]

            chunk_filename = f"segment_{segment_id:04d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)

            frame_filename = f"frame_{segment_id:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            clip_filename = f"clip_{segment_id:04d}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)

            # Calculate frame timestamp
            if frame_position == "start":
                frame_timestamp = segment["start"]
            elif frame_position == "end":
                frame_timestamp = segment["end"]
            else:
                frame_timestamp = (segment["start"] + segment["end"]) / 2

            # Extract components
            self.extract_audio_chunk(audio_path, segment["start"], segment["end"], chunk_path)
            frame_extracted = self.extract_frame(video_path, frame_timestamp, frame_path)
            clip_extracted = self.extract_video_clip(video_path, segment["start"], segment["end"], clip_path)

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
                "frame_timestamp_seconds": frame_timestamp,
                "video_clip_file": clip_filename,
                "video_clip_path": clip_path if clip_extracted else None
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
    description="API for video transcription with audio chunks, frames, and clips extraction"
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
        "message": "Video Transcription API with Frame & Clip Extraction",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "audio": "/audio/{filename} (GET)",
            "frame": "/frame/{filename} (GET)",
            "clip": "/clip/{filename} (GET)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "whisper-tiny",
        "agent_loaded": agent is not None
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

    try:
        # Save uploaded video
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        # Process video
        result = agent.transcribe_video(
            temp_video_path,
            language=language,
            output_dir="audio_chunks",
            frames_dir="video_frames",
            clips_dir="video_clips",
            frame_position=frame_position
        )

        # Clean up temp video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        return JSONResponse(content=result)

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/clip/{filename}")
async def get_video_clip(filename: str):
    """Download a specific video clip (without audio)"""
    clip_path = os.path.join("video_clips", filename)
    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="Video clip not found")
    return FileResponse(clip_path, media_type="video/mp4", filename=filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

