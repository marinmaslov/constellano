# 🌟 Constellano: Star recognition algorithm 🌌 💻

An app written in Python (Flask) that enables you to recognize 👀 constellations on a static picture, using HAAR cascade 🤖.

## 0. Prerequisite ⚙️
To run this app you'll need to install `python 3.10.x`.

## 1. How to run? 🚀
Clone the repo using the following command:
```bash
git clone git@github.com:marinmaslov/constellano.git
```
Position yourself into the constellano directory:
```bash
cd constellano
```
Create a virtual environment:
```bash
python -m venv venv
```
Activate it:
```bash
source venv/bin/activate
```
Install all required modules (make sure you're is the same directory where the requirements.txt file is):
```bash
pip install -r requirements.txt
```
Run the app with the following command (again make sure you're in the same directory as the app.py file):
```bash
flask run
```

## 2. Live version ⚡
A live version of the app can be found <a href="https://constellano.herokuapp.com/">here!</a>

## 3. What's inside? 🧐
A quick look at the apps files and directories.

    .
    ├── constellation-recognition
    |       |── ...
    |       |── images of stars
    |       └── and python code live here (for now, will be displaced)
    ├── static
    |       |── ...
    |       └── static files live here
    ├── templates
    |       |── ...
    |       └── mostly html files live here
    ├── static
    |       |── mostly static files used by Flask app live here
    |       |       |── ...
    |       |       |── ...
    |       |       └── ...
    |       └── ...
    |               |── ...
    |               └── ...
    ├── .gitignore
    ├── Procfile
    ├── README.md
    ├── app.py
    ├── requirements.txt
    └── runtime.txt

## 4. Documentation 📚
The algorithms documentation can be found <a href="https://github.com/marinmaslov/constellano/wiki">here!</a>
### Documentation overview
1. Introduction

## 5. Useful learning materials ✨

Used literature:
#### Recognition and drawing of contours
- https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- https://www.geeksforgeeks.org/white-and-black-dot-detection-using-opencv-python/
- https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/

#### Neural networks
- https://www.codeastar.com/convolutional-neural-network-python/
- https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjIh9vT2IL3AhWY7rsIHa3pAWAQFnoECAUQAQ&url=https%3A%2F%2Fdocs.opencv.org%2F3.4%2Fdb%2Fd28%2Ftutorial_cascade_classifier.html&usg=AOvVaw1DXwOCZiEkt3QQ2Qp5sxK-
- https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjIh9vT2IL3AhWY7rsIHa3pAWAQFnoECAsQAQ&url=https%3A%2F%2Fmedium.com%2Fanalytics-vidhya%2Fhaar-cascades-explained-38210e57970d&usg=AOvVaw3ecM8UKMpA_Qq0CAvOuSdg

#### Using OpenCV (HAAR cascade classifier) with node.js
- https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
> IDEA: Rewrite the whole project in node.js (if possible) and create a React app instead of Flask app
> 
> FURTHER: Create GitHub Wiki page with documentation about the project
> 
> FURTHER-2: Expand the app to be able to connect to Astro-dedicated cameras using ASCOM drivers for live-star-recognition

#### HAAR Cascade Classifier Training:
- https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html

#### Other useful links
- Overall ideas/problems/solutions: https://www.instructables.com/Star-Recognition-Using-Computer-Vision-OpenCV/
- Applying images on contours: https://stackoverflow.com/questions/51606048/python-open-cv-overlay-image-on-contour-of-a-big-image
- Applying PNG's on ROI's:
  - https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
  - https://datahacker.rs/012-blending-and-pasting-images-using-opencv/ 
  - https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
  - https://stackoverflow.com/questions/49737541/merge-two-images-with-alpha-channel 
- Denoising: https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga03aa4189fc3e31dafd638d90de335617
- Contours:
  - https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
  - https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiYu-6ItY73AhVXQ_EDHVEYBncQFnoECAIQAQ&url=https%3A%2F%2Fwww.programcreek.com%2Fpython%2Fexample%2F70455%2Fcv2.drawContours&usg=AOvVaw1iOEPLiG3_O09iqrlr2Ar-
- OpenCV Images Overview: https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html

#### Deployment
- How to deploy Flask App on Heroku: https://stackabuse.com/deploying-a-flask-application-to-heroku/

#### Useful tools
- Diagrams: https://app.diagrams.net
- Design and interaction: https://www.figma.com
  - Figma plugins: https://dev.to/saviomartin/16-must-have-figma-plugins-for-ui-ux-designers-4ffm

> "The real friends of the space voyager are the stars. Their friendly, familiar patterns are constant companions, unchanging, out there." - James Lovell, Apollo Astronaut

<p align="center">
  Flask (Python) App created by Marin Maslov @ <a href="https://www.fesb.unist.hr/">FESB (UNIST)</a>
</p>


DOBRE UPUTE KAKO OD NEGATIVA NAPRAVIT "POYITIVE" TAKO DA UVALIŠ SVOJ TRAŽENI OBJEKT U NJIH: https://memememememememe.me/post/training-haar-cascades/
