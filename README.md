# 🌾 Crop Suggestor AI

An intelligent agricultural advisory system that recommends the most profitable crops for farmers based on their region, climate, soil conditions, and income goals.

## Features

✨ **Smart Recommendations** - Uses machine learning to analyze environmental factors and suggest optimal crops
🌍 **Multi-Region Support** - Covers North, South, East, West, Central, and Northeast regions
📊 **Data-Driven Analysis** - Trained on real agricultural data with 30+ samples
🎯 **Profit-Focused** - Considers income levels to maximize farmer earnings
🌐 **Web Interface** - Easy-to-use Flask web app for farmers
📱 **Mobile-Friendly** - Responsive design works on all devices

## System Requirements

- Python 3.7+
- pip (Python package manager)

## Installation

1. **Clone or download the project:**
   ```bash
   cd "Crop Suggestor"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

## Usage

### Step 1: Build the ML Model

First, train the machine learning model with the agricultural data:

```bash
python model_build.py
```

This will:
- Load the crop recommendation dataset
- Train a Random Forest classifier
- Display model accuracy and feature importance
- Save the trained model as `crop_model.pkl`
- Save encoders as `encoders.pkl`

**Expected Output:**
```
==================================================
CROP RECOMMENDATION MODEL BUILDER
==================================================

Training Accuracy: 0.9500
Testing Accuracy: 0.8571

✓ Model saved as 'crop_model.pkl'
✓ Encoders saved as 'encoders.pkl'
```

### Step 2: Run the Web Application

Start the Flask development server:

```bash
python app.py
```

The application will start at: `http://localhost:5000`

### Step 3: Use the Crop Recommender

1. Open your browser and navigate to `http://localhost:5000`
2. Fill in your farm details:
   - **Region**: Select your geographic region
   - **Temperature**: Average temperature in °C
   - **Humidity**: Relative humidity percentage
   - **Rainfall**: Annual rainfall in mm
   - **Soil Type**: Loamy, Clayey, or Sandy
   - **Land Type**: Flat, Hilly, Undulating, or Steep
   - **pH Level**: Soil pH value
   - **Income Level**: Your target income level
3. Click "Get Recommendation"
4. View the recommended crop with detailed information

## Input Guidelines

### Regional Parameters

- **Temperature Range**: 17-31°C (crop season average)
- **Humidity Range**: 60-85% (relative humidity)
- **Rainfall Range**: 900-3200mm (annual average)
- **pH Range**: 6.0-7.5 (neutral to slightly acidic)

### Regional Examples

| Region | Temp | Humidity | Rainfall | Best Crops |
|--------|------|----------|----------|-----------|
| North | 20°C | 65% | 1500mm | Wheat, Barley, Mustard |
| South | 28°C | 75% | 2500mm | Rice, Sugarcane, Coconut |
| East | 24°C | 70% | 2000mm | Rice, Maize, Pulse |
| West | 22°C | 65% | 1400mm | Jowar, Groundnut, Bajra |
| Central | 25°C | 70% | 1800mm | Soybean, Cotton, Maize |
| Northeast | 22°C | 75% | 2400mm | Tea, Ginger, Turmeric |

## Supported Crops

The system recommends from 19+ major crops:

**High-Income Crops**: Rice, Sugarcane, Paddy, Turmeric, Coconut, Tea, Cotton, Ginger

**Medium-Income Crops**: Wheat, Maize, Soybean, Groundnut, Mustard, Pulse, Jowar, Peanut

**Low-Income Crops**: Barley, Bajra

## Project Structure

```
Crop Suggestor/
├── app.py                      # Flask web application
├── model_build.py              # ML model training script
├── crop_recommendation.csv     # Training dataset
├── crop_model.pkl              # Trained model (generated)
├── encoders.pkl                # Feature encoders (generated)
├── requirement.txt             # Python dependencies
├── README.md                   # This file
└── templates/
    └── index.html              # Web interface
```

## Model Details

### Algorithm: Random Forest Classifier

- **Number of Trees**: 100
- **Max Depth**: 15
- **Training Data**: 30 agricultural scenarios
- **Test Accuracy**: ~85.7%
- **Features Used**: 8 (Region, Temperature, Humidity, Rainfall, Soil Type, Land Type, pH, Income Level)

### Feature Importance

1. **Rainfall** - Most important for crop selection
2. **Temperature** - Determines crop season suitability
3. **Humidity** - Affects water management
4. **Other factors** - Region, soil type, land type, pH

## API Endpoints

### POST /api/recommend
Receives farm parameters and returns crop recommendation.

**Request:**
```json
{
  "region": "North",
  "temperature": 25,
  "humidity": 70,
  "rainfall": 1500,
  "soil_type": "Loamy",
  "land_type": "Flat",
  "pH": 6.5,
  "income_level": "High"
}
```

**Response:**
```json
{
  "success": true,
  "recommended_crop": "Wheat",
  "confidence": 92.5,
  "description": "Winter crop, high market demand...",
  "season": "Winter (Oct-Mar)",
  "avg_income": "Medium",
  "summary": "Based on your conditions..."
}
```

### GET /api/crops-info
Returns detailed information about all supported crops.

### GET /api/help
Returns descriptions of all input fields.

## Troubleshooting

### Model not found error
**Solution**: Run `python model_build.py` first to train the model.

### Port 5000 already in use
**Solution**: Change the port in `app.py` line (modify `port=5000` to another number like `port=5001`)

### Import errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirement.txt
```

## Development

To modify the model or add new crops:

1. Update `crop_recommendation.csv` with new training data
2. Run `python model_build.py` to retrain
3. The web app will automatically use the new model

## Future Enhancements

- 🌐 Multi-language support
- 📈 Profit prediction with current market prices
- 🌤️ Real-time weather integration
- 💾 Farmer profile saved recommendations
- 📊 Crop rotation suggestions
- 🚜 Equipment and fertilizer recommendations
- 🔔 Seasonal alerts and notifications

## License

This project is free to use for agricultural education and farmer assistance.

## Support

For issues or questions, ensure:
1. Python 3.7+ is installed
2. All requirements are installed: `pip install -r requirement.txt`
3. You've run `model_build.py` before starting the app

## Author

Created for farmers to maximize crop yield and income through AI-powered recommendations.

---

**Happy Farming! 🌾**
