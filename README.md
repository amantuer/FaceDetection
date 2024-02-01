# Face Detection and Embedding Project

## Overview
This project includes a Python script that performs face detection on an image, generates embeddings for each detected face, and then stores these embeddings in a PostgreSQL database. It's a basic demonstration of integrating computer vision and database operations for face recognition purposes.

## Prerequisites
Before running this script, ensure you have the following installed:
- Python 3.x
- OpenCV-Python
- PIL (or Pillow)
- NumPy
- psycopg2
- PostgreSQL database server
- `imgbeddings` library (ensure this is installed or accessible in your environment)

Also, set up a PostgreSQL database and create a table named `pictures` with the appropriate schema.

## Installation
Clone this repository to your local machine using:

```bash
git clone [URL to the repository]
```

Install the required Python libraries using:

```bash
pip install numpy opencv-python pillow psycopg2
```

(Replace with the correct command if your setup requires a different way to install Python packages.)

## Configuration
Before running the script, make sure to update the following in the `main.py` file:
- The path to the image file you want to process.
- The path to the Haar Cascade XML file for face detection (`haarcascade_frontalface_default.xml`).
- The database connection string for your PostgreSQL database.

## Running the Script
To run the script, navigate to the directory containing `main.py` and run:

```bash
python main.py
```

The script will process the specified image, detect faces, generate embeddings, and store these embeddings in your PostgreSQL database.

## Troubleshooting
Ensure all dependencies are correctly installed and that the PostgreSQL server is running and accessible. Check the paths to the image and Haar Cascade file in the script.

## License

This project is licensed under the [LICENSE NAME] License - see the [LICENSE.md](LICENSE.md) file for details.



