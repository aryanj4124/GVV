from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__, template_folder='/home/aryan/GVV/audio_player/frontend')

# Directory where your audio files are stored
AUDIO_DIR = '/home/aryan/GVV/audio_player/songs'

# List of audio files (replace these with your own audio files)
audio_files = os.listdir(AUDIO_DIR)

# Initialize the currently playing audio file
current_audio = None

@app.route('/')
def index():
    return render_template('front.html', audio_files=audio_files, current_audio=current_audio)

@app.route('/play/<audio_file>')
def play_audio(audio_file):
    global current_audio
    current_audio = audio_file
    return '', 204  # No content, used for AJAX requests

@app.route('/stop')
def stop_audio():
    global current_audio
    current_audio = None
    return '', 204  # No content, used for AJAX requests

@app.route('/songs/<path:filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)

