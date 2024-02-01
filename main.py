import cv2
import numpy as np
from PIL import Image
import psycopg2
import os
from imgbeddings import imgbeddings #make sure this python library is installed

def detect_faces(file_name, haar_cascade_file):
    if not os.path.exists(file_name):
        print(f"Error: File {file_name} not found.")
        return []

    img = cv2.imread(file_name, 0)
    if img is None:
        print(f"Error: Unable to read the image {file_name}.")
        return []

    haar_cascade = cv2.CascadeClassifier(haar_cascade_file)
    faces = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

    if len(faces) == 0:
        print("No faces detected.")
        return []

    face_images = []
    for x, y, w, h in faces:
        cropped_face = img[y : y + h, x : x + w]
        face_images.append(cropped_face)

    return face_images

def generate_embedding(image):
    ibed = imgbeddings()  # Ensure this is correctly set up
    return ibed.to_embeddings(image)

def store_embeddings_to_db(face_images, db_connection_string):
    conn = psycopg2.connect(db_connection_string)
    cur = conn.cursor()

    for face in face_images:
        face_image = Image.fromarray(face)
        embedding = generate_embedding(face_image)
        string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
        cur.execute("INSERT INTO pictures (embedding) VALUES (%s)", (string_representation,))

    conn.commit()
    cur.close()
    conn.close()

def find_most_similar_face(embedding, db_connection_string):
    conn = psycopg2.connect(db_connection_string)
    cur = conn.cursor()

    string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
    cur.execute("SELECT filename FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    if row:
        return row[0]
    else:
        return "No similar faces found."

def main():
    haar_cascade_file = "haarcascade_frontalface_default.xml"
    db_connection_string = "postgres://avnadmin:AVNS_RaIabVoQwxxuMQmKVWR@pg-31120d93-faces-db.a.aivencloud.com:16155/defaultdb?sslmode=require"  # Replace with your database connection string

    # Process the first image for learning and storing faces
    learning_image_file = "musk_friends.png"  # Replace with your learning image file
    face_images = detect_faces(learning_image_file, haar_cascade_file)
    if face_images:
        store_embeddings_to_db(face_images, db_connection_string)
        print("Faces from the learning image have been processed and stored.")

    # Process a new image for comparison
    new_image_file = "musk.png"  # Replace with your new image file
    new_face_images = detect_faces(new_image_file, haar_cascade_file)
    if new_face_images:
        for face in new_face_images:
            face_image = Image.fromarray(face)
            embedding = generate_embedding(face_image)
            similar_face_file = find_most_similar_face(embedding, db_connection_string)
            print(f"The most similar face is in file: {similar_face_file}")

if __name__ == "__main__":
    main()
