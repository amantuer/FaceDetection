import cv2
import numpy as np
from PIL import Image
import psycopg2
import os
from imgbeddings import imgbeddings  # Make sure this library is installed or accessible

# Function to detect faces and save them
def detect_and_save_faces(file_name, haar_cascade_file):
    if not os.path.exists(file_name):
        print(f"Error: File {file_name} not found.")
        return []

    img = cv2.imread(file_name, 0)
    if img is None:
        print(f"Error: Unable to read the image {file_name}.")
        return []

    haar_cascade = cv2.CascadeClassifier(haar_cascade_file)
    faces = haar_cascade.detectMultiScale(
        img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
    )

    if len(faces) == 0:
        print("No faces detected.")
        return []

    if not os.path.exists('stored-faces'):
        os.makedirs('stored-faces')

    face_files = []
    for i, (x, y, w, h) in enumerate(faces):
        cropped_image = img[y: y + h, x: x + w]
        target_file_name = f'stored-faces/face_{i}.jpg'
        cv2.imwrite(target_file_name, cropped_image)
        face_files.append(target_file_name)

    return face_files

# Function to generate embeddings and insert into database
def generate_embeddings_and_insert(face_files, db_connection_string):
    conn = psycopg2.connect(db_connection_string)
    cur = conn.cursor()

    ibed = imgbeddings()  # Ensure this is correctly set up

    for filename in face_files:
        try:
            img = Image.open(filename)
            embedding = ibed.to_embeddings(img)
            string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
            cur.execute("INSERT INTO pictures values (%s, %s)", (filename, string_representation))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    conn.commit()
    cur.close()
    conn.close()

# Main execution
def main():
    # Replace with your image file and Haar Cascade path
    file_name = "musk_friends.png"
    haar_cascade_file = "haarcascade_frontalface_default.xml"

    # Replace with your database connection string
    db_connection_string = "<SERVICE_URI>"

    face_files = detect_and_save_faces(file_name, haar_cascade_file)
    if face_files:
        generate_embeddings_and_insert(face_files, db_connection_string)

if __name__ == "__main__":
    main()
