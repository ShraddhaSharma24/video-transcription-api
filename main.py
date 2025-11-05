import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from datetime import timedelta
from PIL import Image

class VideoTranscriptionAgent:
    def __init__(self, model_size="tiny"):
        self.model = whisper.load_model(model_size)

    def extract_audio(self, video_path, audio_output_path="temp_audio.wav"):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path, logger=None)
        video.close()
        return audio_output_path

    def extract_frame(self, video_path, timestamp, output_path):
        video = VideoFileClip(video_path)
        frame = video.get_frame(timestamp)
        video.close()
        img = Image.fromarray(frame)
        img.save(output_path)
        return output_path

    def extract_video_clip(self, video_path, start_time, end_time, output_path):
        video = VideoFileClip(video_path)
        clip = video.subclip(start_time, end_time).without_audio()
        clip.write_videofile(
            output_path,
            codec='libx264',
            audio=False,
            logger=None,
            preset='ultrafast'
        )
        video.close()
        clip.close()
        return output_path

    def format_timestamp(self, seconds):
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def extract_audio_chunk(self, audio_path, start_time, end_time, output_path):
        audio = AudioSegment.from_wav(audio_path)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        chunk = audio[start_ms:end_ms]
        chunk.export(output_path, format="wav")
        return output_path

    def transcribe_video(self, video_path, language=None, output_dir="audio_chunks",
                        frames_dir="video_frames", clips_dir="video_clips",
                        frame_position="middle"):

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)

        audio_path = self.extract_audio(video_path)

        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )

        formatted_result = {
            "full_text": result["text"],
            "language": result["language"],
            "segments": []
        }

        for segment in result["segments"]:
            segment_id = segment["id"]
            chunk_filename = f"segment_{segment_id:04d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)

            frame_filename = f"frame_{segment_id:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            clip_filename = f"clip_{segment_id:04d}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)

            if frame_position == "start":
                frame_timestamp = segment["start"]
            elif frame_position == "end":
                frame_timestamp = segment["end"]
            else:
                frame_timestamp = (segment["start"] + segment["end"]) / 2

            self.extract_audio_chunk(audio_path, segment["start"], segment["end"], chunk_path)
            self.extract_frame(video_path, frame_timestamp, frame_path)
            self.extract_video_clip(video_path, segment["start"], segment["end"], clip_path)

            formatted_segment = {
                "id": segment["id"],
                "start": self.format_timestamp(segment["start"]),
                "end": self.format_timestamp(segment["end"]),
                "start_seconds": segment["start"],
                "end_seconds": segment["end"],
                "duration_seconds": segment["end"] - segment["start"],
                "text": segment["text"].strip(),
                "audio_file": chunk_filename,
                "frame_file": frame_filename,
                "video_clip_file": clip_filename,
            }

            formatted_result["segments"].append(formatted_segment)

        os.remove(audio_path)
        return formatted_result

app = FastAPI(title="Video Transcription API")
agent = VideoTranscriptionAgent(model_size="tiny")

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(video: UploadFile = File(...), language: str = None, frame_position: str = "middle"):
        temp_video_path = f"temp_{video.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
        try:
            result = agent.transcribe_video(temp_video_path, language=language, frame_position=frame_position)
            os.remove(temp_video_path)
            return JSONResponse(content=result)
        except Exception as e:
            os.remove(temp_video_path)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return FileResponse(os.path.join("audio_chunks", filename))

@app.get("/frame/{filename}")
async def get_frame(filename: str):
    return FileResponse(os.path.join("video_frames", filename))

@app.get("/clip/{filename}")
async def get_clip(filename: str):
    return FileResponse(os.path.join("video_clips", filename))
