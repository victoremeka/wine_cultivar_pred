#!/bin/bash

# Wine Cultivar Prediction System - Setup Script
# This script sets up the environment and prepares the application for deployment

echo "🍷 Setting up Wine Cultivar Prediction System..."
echo "================================================"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p model

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To run the application locally:"
echo "   streamlit run app.py"
echo ""
echo "📊 To train the model:"
echo "   Open model/model_building.ipynb in Jupyter and run all cells"
echo ""
