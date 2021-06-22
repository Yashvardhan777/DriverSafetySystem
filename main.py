from flask import Flask, render_template,Response
from drowsiness_detection import Detector

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(drowsiness_detection):
    while True:
        image = drowsiness_detection.detection()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(image) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Detector()),
                mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()