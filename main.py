from flask import Flask,render_template,Response,request
import cv2
import numpy as np
import dlib
from imutils import face_utils

app=Flask("drowsiness_detection")

global start,cap
start=0
def EucledianDistance(X,Y):
    return np.linalg.norm(X-Y)

def blinked(a,b,c,d,e,f):
    up=EucledianDistance(b,d)+EucledianDistance(c,e)
    down=EucledianDistance(a,f)
    ratio=up/(2.0*down)
    if ratio>0.25:
        return 2
    elif ratio>0.21 and ratio<=0.25:
        return 1
    else:
        return 0
# cap=cv2.VideoCapture(0)
def predict():
    global start,cap
    if start==1:
        cap=cv2.VideoCapture(0)
    else:
        return

    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor('model_file/shape_predictor_68_face_landmarks.dat')

    sleep=drowsy=active=0
    status=""
    color=(0,0,0)
    while start==1:
        _,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces=detector(gray)
        for face in faces:            
            landmarks=predictor(gray,face)
            landmarks=face_utils.shape_to_np(landmarks)
            
            left_blink=blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
            right_blink=blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])
            
            if left_blink==0 or right_blink==0:
                sleep+=1
                drowsy=0
                active=0
                if(sleep>=5):
                    status="SLEEPING !!!"
                    color=(255,0,0)
            elif left_blink==1 or right_blink==1:
                sleep=0
                drowsy+=1
                active=0
                if(drowsy>=5):
                    status="DROWSY !!"
                    color=(0,0,255)
            else:
                sleep=0
                drowsy=0
                active+=1
                if(active>=5):
                    status="Active"
                    color=(0,255,0)
                    
            cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(predict(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/requests',methods=['POST'])
def tasks():
    global start,cap
    if request.form.get('stop') == 'Stop/Start Webcam':
        if(start==1):
            start=0
            cap.release()
            cv2.destroyAllWindows()                
        else:
            start=1
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)