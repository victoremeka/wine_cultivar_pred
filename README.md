# Wine Cultivar Origin Prediction System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning web application that predicts wine cultivar (origin/class) based on chemical properties using the UCI Wine Dataset.

## 👤 Student Information

- **Name:** Victor Emeka
- **Matric Number:** 23cg034065
- **Algorithm:** Random Forest Classifier
- **Model Persistence:** Joblib

## 📊 Project Overview

This project implements a multiclass classification system to predict wine cultivars based on their chemical properties. The system uses a Random Forest Classifier trained on 6 selected chemical features from the Wine Dataset.

### Features Used (6 out of 13 available):
1. **Alcohol** - Alcohol content (%)
2. **Malic Acid** - Malic acid concentration (g/L)
3. **Ash** - Ash content (g/L)
4. **Total Phenols** - Total phenolic compounds (mg/L)
5. **Flavanoids** - Flavanoid content (mg/L)
6. **Color Intensity** - Wine color intensity (units)

### Target Variable:
- **Cultivar** - Wine cultivar class (0, 1, or 2)

## 🏗️ Project Structure

```
WineCultivar_Project_victoremeka_23cg034065/
│
├── app.py                              # Streamlit web application
├── requirements.txt                     # Python dependencies
├── setup.sh                            # Setup script
├── README.md                           # Project documentation
├── WineCultivar_hosted_webGUI_link.txt # Deployment information
│
└── model/
    ├── model_building.ipynb            # Jupyter notebook for model training
    ├── wine_cultivar_model.pkl         # Trained model (generated after training)
    └── scaler.pkl                      # Feature scaler (generated after training)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd WineCultivar_Project_victoremeka_23cg034065
   ```

2. **Run the setup script (Linux/Mac)**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   **Or install manually (Windows/Any OS)**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   - Open `model/model_building.ipynb` in Jupyter Notebook or Jupyter Lab
   - Run all cells to train the model and generate:
     - `wine_cultivar_model.pkl`
     - `scaler.pkl`

4. **Run the web application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and navigate to: `http://localhost:8501`

## 📈 Model Development (PART A)

### Dataset
- **Source:** UCI Machine Learning Repository / sklearn.datasets
- **Total Samples:** 178
- **Classes:** 3 wine cultivars
- **Features:** 13 chemical properties (6 selected for this project)

### Data Preprocessing
1. **Missing Values:** Checked and handled (no missing values in Wine dataset)
2. **Feature Selection:** Selected 6 relevant chemical features
3. **Feature Scaling:** Applied StandardScaler (mandatory due to varying feature ranges)
4. **Train-Test Split:** 80% training, 20% testing with stratification

### Model Training
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42

### Model Evaluation
The model is evaluated using the following multiclass classification metrics:
- **Accuracy**
- **Precision** (weighted and macro averages)
- **Recall** (weighted and macro averages)
- **F1-Score** (weighted and macro averages)
- **Classification Report** (per-class metrics)
- **Confusion Matrix**

### Model Persistence
- **Method:** Joblib
- **Files:** 
  - `wine_cultivar_model.pkl` - Trained Random Forest model
  - `scaler.pkl` - Fitted StandardScaler

## 🌐 Web Application (PART B)

### Technology Stack
- **Framework:** Streamlit
- **Frontend:** HTML/CSS (embedded in Streamlit)
- **Backend:** Python with scikit-learn

### Features
- 🔬 Interactive input form for wine chemical properties
- 🎯 Real-time cultivar prediction
- 📊 Prediction probability distribution visualization
- 💡 Sample data loading for testing
- 📱 Responsive design with modern UI
- ℹ️ Comprehensive information sidebar

### User Interface
The web application provides:
1. Input fields for all 6 chemical features
2. Prediction button to classify wine cultivar
3. Visual display of predicted cultivar with confidence level
4. Probability distribution chart
5. Input summary review

## 📤 GitHub Repository (PART C)

Repository includes:
- Complete source code
- Model training notebook
- Web application
- Configuration files
- Documentation

## 🌍 Deployment (PART D)

### Supported Platforms
- ✅ Streamlit Cloud (Recommended)
- ✅ Render.com
- ✅ PythonAnywhere.com
- ✅ Vercel
- ✅ Scorac.com

### Streamlit Cloud Deployment (Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Wine Cultivar Prediction System"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

### Render.com Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Deploy

## 🔧 Local Development

### Running Jupyter Notebook
```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open model/model_building.ipynb
```

### Running Streamlit App
```bash
# Run with default settings
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8080

# Run without watching for file changes
streamlit run app.py --server.runOnSave false
```

## 📊 Model Performance

Expected performance metrics (after training):
- **Accuracy:** ~95-98%
- **Precision (weighted):** ~96-99%
- **Recall (weighted):** ~95-98%
- **F1-Score (weighted):** ~96-98%

*Note: Actual values will be displayed after running the model_building.ipynb notebook*

## 🔬 Technical Details

### Feature Scaling
StandardScaler is used to normalize features:
- **Mean:** 0
- **Standard Deviation:** 1
- **Purpose:** Handle varying feature ranges for optimal model performance

### Random Forest Classifier
Ensemble learning method that:
- Constructs multiple decision trees
- Outputs the mode of classes (classification)
- Reduces overfitting through averaging
- Provides feature importance rankings

## 📝 Assignment Requirements Checklist

### Part A - Model Development ✅
- [x] Load Wine dataset
- [x] Data preprocessing
- [x] Handle missing values
- [x] Feature selection (6 features)
- [x] Feature scaling (StandardScaler)
- [x] Implement Random Forest Classifier
- [x] Train the model
- [x] Evaluate with multiclass metrics
- [x] Save model using Joblib

### Part B - Web GUI ✅
- [x] Load saved model
- [x] User input interface
- [x] Pass data to model
- [x] Display predictions
- [x] Use Streamlit framework

### Part C - GitHub Structure ✅
- [x] Proper directory structure
- [x] app.py
- [x] requirements.txt
- [x] model/model_building.ipynb
- [x] model/wine_cultivar_model.pkl
- [x] Documentation files

### Part D - Deployment ✅
- [x] Deployment instructions
- [x] WineCultivar_hosted_webGUI_link.txt

## 🆘 Troubleshooting

### Model files not found
- Make sure you've run all cells in `model/model_building.ipynb`
- Check that `wine_cultivar_model.pkl` and `scaler.pkl` exist in the `model/` directory

### Package installation errors
- Upgrade pip: `pip install --upgrade pip`
- Use Python 3.8+: `python --version`
- Install packages individually if needed

### Streamlit not running
- Check if port 8501 is available
- Try a different port: `streamlit run app.py --server.port 8080`

## 📚 Dependencies

- **streamlit** - Web application framework
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **joblib** - Model persistence
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization

## 📄 License

This project is submitted as part of academic coursework.

## 👨‍💻 Author

**Victor Emeka**  
Matric Number: 23cg034065  

---

**Submission Date:** January 21, 2026  
**Project:** Wine Cultivar Origin Prediction System  
**Course:** Machine Learning  
