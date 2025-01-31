from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import subprocess
import whisper  # Import Whisper for transcription
from pydub import AudioSegment
from pydub.silence import split_on_silence
from colorama import Fore, init

init(autoreset=True)

app = Flask(__name__)  # Fixed _name to __name__
CORS(app)  # Allow cross-origin requests

# Create a directory to save the audio files if it doesn't exist
AUDIO_DIR = 'uploaded_audio'
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Load the Whisper model globally to avoid loading it multiple times
model = whisper.load_model('medium')  # You can choose 'tiny', 'base', 'small', 'medium', 'large' depending on performance

def split_audio_into_chunks(audio_path, output_folder, min_silence_len=700, silence_thresh=-40):
    """Split audio into smaller chunks based on silence."""
    print(Fore.CYAN + "Splitting audio into chunks...")
    try:
        sound = AudioSegment.from_file(audio_path)  # This handles different formats like webm, mp3
        chunks = split_on_silence(
            sound,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_folder, f"chunk{i + 1}.wav")
            chunk.export(chunk_file, format="wav")
            chunk_files.append(chunk_file)
            print(Fore.GREEN + f"Exported: {chunk_file}")

        print(Fore.CYAN + f"Total {len(chunk_files)} chunks created.")
        return chunk_files
    except Exception as e:
        print(Fore.RED + f"Error during audio splitting: {e}")
        return []

def transcribe_audio(audio_file_path):
    """Function to transcribe an audio file using Whisper."""
    try:
        print(f"Transcribing audio file: {audio_file_path}")  # Debugging: Check the file path
        
        result = model.transcribe(audio_file_path, task="translate")  # Automatically translate audio to English
        
        print(f"Transcription result: {result['text']}")  # Debugging: Log the transcription output
        
        return result['text']  # Return the transcribed text
    except Exception as e:
        print(f"Error during transcription: {str(e)}")  # Debugging: Print the error
        return f"Error transcribing audio: {str(e)}"


@app.route('/')
def index():
    return 'Audio upload server is running. Please use the frontend to upload audio.'

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    audio_file = request.files['audio']
    custom_filename = request.form.get('filename', 'recording')

    # Save the audio file temporarily
    timestamp = int(time.time())
    webm_file_path = os.path.join(AUDIO_DIR, f'{custom_filename}_{timestamp}.webm')
    audio_file.save(webm_file_path)

    # Convert webm to mp3 (assumed you're using FFmpeg as in previous code)
    mp3_file_path = os.path.join(AUDIO_DIR, f'{custom_filename}_{timestamp}.mp3')
    try:
        subprocess.run(['ffmpeg', '-i', webm_file_path, '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', mp3_file_path], check=True)
        os.remove(webm_file_path)  # Remove the original webm file after conversion
        
        # Split the audio file into chunks for better processing
        chunks_folder = os.path.join(AUDIO_DIR, f"chunks_{timestamp}")
        chunk_files = split_audio_into_chunks(mp3_file_path, chunks_folder)
        
        if chunk_files:
            transcription_text = ""
            for chunk_file in chunk_files:
                transcription_text += transcribe_audio(chunk_file) + "\n"
            
            # Save final transcription
            transcription_file_path = os.path.join(AUDIO_DIR, f'{custom_filename}_{timestamp}_transcription.txt')
            with open(transcription_file_path, "w", encoding="utf-8") as file:
                file.write(transcription_text)
            
            return jsonify({
                "message": "Audio uploaded, converted, and transcribed successfully",
                "audioFilename": f'{custom_filename}_{timestamp}.mp3',
                "transcription": transcription_text
            }), 200
        
        else:
            return jsonify({"error": "Failed to split audio into chunks"}), 500

    except Exception as e:
        return jsonify({"error": f"Error saving or converting audio: {str(e)}"}), 500


if __name__ == '__main__':  # Fixed _name to __name__
    app.run(debug=True)  # Ensure Flask starts with debugging mode
