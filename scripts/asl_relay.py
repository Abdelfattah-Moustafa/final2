from flask import Flask
from flask_socketio import SocketIO, emit
import logging

# Disable distracting logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", max_decode_size=10**7) # Support large video frames

@socketio.on('connect')
def handle_connect():
    print("Device connected to ASL-Relay.")

@socketio.on('video_frame')
def handle_video(data):
    # Broadcast video to everyone EXCEPT the sender
    emit('remote_video', data, broadcast=True, include_self=False)

@socketio.on('translate_sentence')
def handle_sentence(data):
    # Broadcast the translated text (Sign-to-Text result)
    print(f"Relaying Sentence: {data}")
    emit('remote_sentence', data, broadcast=True, include_self=False)

if __name__ == '__main__':
    print("\n" + "="*40)
    print("   ASL-CONNECT RELAY SERVER: RUNNING   ")
    print("   Listening on: http://localhost:5000 ")
    print("="*40 + "\n")
    socketio.run(app, port=5000, debug=False)
