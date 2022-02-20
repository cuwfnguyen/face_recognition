import time
import jsonpickle
import tensorflow as tf
import keras
import flask
from flask import Flask, request, Response
from utils import *
from flask_sqlalchemy import SQLAlchemy


print("cv2_version : {}".format(cv2.__version__))
print("tf_version : {}".format(tf.__version__))
print("keras_version : {}".format(keras.__version__))
print("flask_version : {}".format(flask.__version__))

BASE_DIR = "../Server"


# Initialize the Flask application
app = Flask(__name__)

# adding configuration for using a sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database2.db'

# Creating an SQLAlchemy instance
db = SQLAlchemy(app)


# Models
class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=False, nullable=False)
    photo_url = db.Column(db.String(200), unique=False, nullable=False)

    def __repr__(self):
        return f"Name : {self.name}"


# create db
db.create_all()
db.session.commit()

rec = None
face_db = None
rec_data = None
save_path = None


@app.route("/create", methods=['GET', 'POST'])
def create_face_image():
    name = request.json["name"]
    image_bytes = bytes(request.json["image"], 'ISO-8859-1')
    np_array_img = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(np_array_img, cv2.IMREAD_COLOR)
    if img is not None:
        path = "{}/db/{}.png".format(BASE_DIR, name)
        cv2.imwrite(path, img)
        # save on database
        user = UserProfile(name=name, photo_url=path)
        db.session.add(user)
        db.session.commit()

        response = {'message': 'Successfully added!'}
        status_code = 200

    else:
        response = {'message': 'Add failed!'}
        status_code = 404
    print(response)
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=status_code, mimetype="application/json")


@app.route("/face-recognize", methods=['POST'])
def face_recognize():
    db_path = os.path.join(BASE_DIR, "db")
    print("db_path", db_path)
    # get list file from database
    user_queryset = UserProfile.query.all()
    print("user_queryset", user_queryset)
    list_photo = []
    for user in user_queryset:
        list_photo.append(user.photo_url)

    face_db = FaceDB()
    if len(list_photo) != 0:
        print("list_photo", list_photo)
        face_db.load_from_db(list_photo, rec)
    else:
        face_db.load(db_path, rec)
    db_f_count = len(face_db.get_data())
    print("Face DB loaded: " + str(db_f_count))

    print("Processing recognition request... ")
    t1 = time.time()
    np_array_img = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(np_array_img, cv2.IMREAD_COLOR)
    embds = rec.embeddings(img)
    data = rec.recognize(embds, face_db)
    t2 = time.time()
    dt = t2-t1
    print("Recognition request processed: " + str(dt) + " sec")
    
    rec_data.count()
    if data is not None:
        (name, dist, p_photo) = data
        conf = 1 - dist
        percent = int(conf*100)
        ps = ("%03d" % rec_data.get_count()) + "_" + name + "_" + ("%03d" % percent) + ".png"
        response = {
            "message": "RECOGNIZED",
            "name": name,
            "percent": str(percent)}
    else:
        ps = ("%03d" % rec_data.get_count()) + "_unrecognized" + ".png"
        response = {"message": "UNRECOGNIZED"}

    if save_path is not None:
       ps = os.path.join(save_path, ps)
       cv2.imwrite(ps, img)
        
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    # host = str(sys.argv[1])
    # port = int(sys.argv[2])

    # FaceNet recognizer
    m_file = r"./models/facenet_keras.h5"
    rec = FaceNetRec(m_file, 0.35)
    rec_data = RecData()
    print("Recognizer loaded.")
    print(rec.get_model().inputs)
    print(rec.get_model().outputs)
    
    # Face DB 
    save_path = r"../rec"
    # db_path = r"E:\20211\DATK\1.1Linhduong\face-recognize-with-flask-master\Server\db"
    
    print("Face recognition running")
          
    host = "127.0.0.1"
    port = 5000
    app.run(host=host, port=port, threaded=False, debug=True)

# END
