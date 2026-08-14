
'''a function that store the face data and landmarks along with lable and store
   the data as an numpy array in the database'''
import pickle
import sqlite3
import numpy as np

def init_db():
    
    db=sqlite3.connect("face_data.db")
    cursor=db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS face_data(face_landmarks BLOB,label TEXT,name TEXT,Cluster_name TEXT)")
    db.commit()
    db.close()

def find_face_data(face_landmarks):
    init_db()
    db=sqlite3.connect("face_data.db")
    cursor=db.cursor()
    cursor.execute("SELECT * FROM face_data")
    face_data=cursor.fetchall()
    db.close()
    
    for face_data in face_data:
        face_landmarks=pickle.loads(face_data[0])
        label=face_data[1]
        name=face_data[2]
        Cluster_name=face_data[3]
        # print(face_landmarks)
        print(label)
        print(name)
        print(Cluster_name)
    return face_data

    
def save_face_data(face_landmarks,label,name,cluster_name):
    
    init_db()
    face_landmarks=np.array(face_landmarks)
    # print(face_landmarks)
    face_landmarks=pickle.dumps(face_landmarks)
    db=sqlite3.connect("face_data.db")
    cursor=db.cursor()
    cursor.execute("INSERT INTO face_data(face_landmarks,label,name) VALUES (?,?,?)",(face_landmarks,label,name))
    db.commit()
    db.close()



        