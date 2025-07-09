## Lunar Crater and Boulder Detection

This project uses a YOLOv8 object detection model to identify craters and boulders on lunar surface images, helping to determine safe landing zones. It includes a web interface for uploading images and visualizing detection results.


# DATASET DETAILS

- DATASET: OBJECT DETECTION WITH LABELS.TXT
- DETAILS: LUNAR SURFACE CRATERS AND BOULDERS DETECTION
- LINK TO DATASET: [Dataset(Google Drive)](https://drive.google.com/drive/folders/1MYrhCtq5oQPsNDDOUdGTkW_H1VF8yXzw?usp=sharing)
- STRUCTURE: Dataset
            ├───train
            │   ├───images
            │   └───labels
            └───valid
                ├───images
                └───labels

<!-- labels.txt format -->
<class_id> <x_center> <y_center> <width> <height>
            
# MODEL DETAILS 

- ARCHITECTURE: YOLOv8l.pt
- FRAMEWORK: ULTRALYTICS
- EPOCHS: 100
- INPUT SIZE: 640x640 px
- TRAINED MODEL: best.pt(saved in model/best.pt)
- NUMBER OF CLASSES: 2['craters','boulders']
- REQUIREMENTS: python 3.10.0 
- GPU: NVIDIA RTX 4060 8GB
- CUDA VERSION: 12.1
- OUTPUTS - Boxes around craters and boulders, along with safest landing zone. Also generates GRAD-CAM Heatmap and an informative table.
<!-- older python version 3.10.0 was used due to limited support of cuda 12.1 to python 3.10-->

--- IT IS HIGHLY RECOMMENDED TO DOWNLOAD PYTHON 3.10.0 AND ADD TO PATH FOR PROPER FUNCTIONING OF THE MODEL.
--- [DOWNLOAD PYTHON 3.10.0](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)


# INFORMATION ABOUT SUBMISSION FOLDER

- main.py: file containing backend code written in python
- static>index.html: file containing frontend for webpage in html language
- requirements.txt: containing all the python libraries along with versions used in project
- model>best.pt: it is the model trained using yolov8l for detecting lunar craters and boulders 
- report.pdf : Report containing information about full project like approach behind this, challenges faced and creativity
- structure: Submission
             ├───model
             ├───Results
             │   ├───images
             │   └───labels
             └───static

# HOW TO RUN

--- IT IS HIGHLY RECOMMENDED TO DOWNLOAD PYTHON 3.10.0 AND ADD TO PATH FOR PROPER FUNCTIONING OF THE MODEL.
--- [DOWNLOAD PYTHON 3.10.0](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)

<!-- Creating and Activating virtual Environment in Windows -->
1. Open Windows Powershell.
2. Paste the lines given below in Powershell.
    cd <project_folder_path>
    py -3.10 -m venv <virtual_env_name>   #paste as-is, since different python function may exist on your device
    ./<virtual_env_name>/Scripts/Activate    
    
## if you are getting POWERSHELL EXECUTION POLICY ERROR after this. Paste this in terminal for bypassing it for current sessions,  (Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned) -->
<!---- If environment is activated, term will appear like -> (<virtual_env_name>) PS <folder path> ---->

<!-- NOTE: Do every step in virtual environment -->


<!---Install all the required libraries--->
3. Paste in Powershell to download required libraries.
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt


<!-- Starting the backend of website -->
4. Paste in Powershell to start backend.
    uvicorn main:app --reload

5. Check http://127.0.0.1:8000 on browser, it should show (lunar detection backend is up), confirming backend is running perfectly.

<!-- Starting the frontend  -->
6. Open and paste in Powershell to start frontend.

    cd <project_folder_path>
    ./<virtual_env_name>/Scripts/Activate
    cd static
    python -m http.server 5000

7. Go to localhost:5000 on browser, it will locally show the frontend on browser.

8. You can use the frontend to upload images and get detection results.

9. Logs will appear on the backend Powershell. 