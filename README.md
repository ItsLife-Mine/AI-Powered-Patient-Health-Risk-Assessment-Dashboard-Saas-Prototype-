AI-Powered Patient Health Risk Assessment Dashboard (SaaS Prototype)
This repository contains the foundational code for a Minimum Viable Product (MVP) of a SaaS application aimed at enhancing preventative care in small healthcare clinics. This project showcases the integration of Machine Learning (ML) in a Healthcare IT context, serving predictions through a production-ready API and displaying results on an interactive dashboard.

Key Features & Project Goals
Predictive Model: Utilizes a Logistic Regression classifier trained on public health data (Pima Indians Diabetes Dataset) to assess a patient's risk of a chronic condition (e.g., Type 2 Diabetes).

Decoupled Architecture: Features a backend Flask REST API (app.py) to serve predictions, demonstrating experience with API development and model deployment.

Client-Facing Dashboard: A simple HTML/JavaScript frontend (index.html) that consumes the API, providing a clear, color-coded risk assessment score, demonstrating competence in building user-friendly Dashboards.

SaaS Positioning: The prototype simulates a commercial tool, highlighting a SaaS product mindset and focus on user value (clinic efficiency, improved patient outcomes).

Technology Stack & Skills Demonstrated
Category	Technologies Used	Skills Demonstrated
Machine Learning	Python, scikit-learn, joblib	AI/ML Concepts
Backend/API	Flask, Flask-CORS	API Experience, SaaS Products
Data Handling	pandas, Public Health Datasets	Healthcare IT
Frontend	HTML5, CSS, JavaScript (Fetch API)	Dashboards, Software Development
Design/Presentation	Implied: Clean UI/UX for Client-Facing Roles	Canva/Figma (for planning/pitch deck)

Export to Sheets

Setup and Local Installation
To run the full application (model trainer, API, and frontend dashboard) locally, follow these steps.

1. Prerequisites
Python 3.x

Git

2. Clone the Repository
Bash

git clone [Your Repository URL]
cd [project-folder-name]
3. Set up the Environment
It is strongly recommended to use a virtual environment:

Bash

# Create the virtual environment
python -m venv venv

# Activate the environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
4. Install Dependencies
Install all required Python libraries:

Bash

pip install pandas scikit-learn flask flask-cors joblib
5. Execute the Project (Three Steps)
Step A: Train and Save the Model
This script trains the model and saves the necessary files (model.pkl, scaler.pkl) required by the API.

Bash

python train_model.py
Step B: Run the Prediction API
Keep this terminal window open. The server will run on http://127.0.0.1:5000/.

Bash

python app.py
Step C: Launch the Dashboard
Open the frontend file in your web browser.

Locate the index.html file in your project folder.

Double-click the file to open it in Chrome, Firefox, or any other browser.

The dashboard will now connect to the running API and provide real-time risk predictions based on the input data.

Project Structure
.
├── train_model.py      # ML model training and saving script.
├── app.py              # Flask API for serving model predictions.
├── index.html          # Frontend Dashboard (HTML, CSS, JS).
├── model.pkl           # Saved trained Logistic Regression model.
├── scaler.pkl          # Saved StandardScaler object for data processing.
└── README.md           # This file.
