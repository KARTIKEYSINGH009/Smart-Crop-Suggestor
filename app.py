# Crop Recommendation System for Indian Farmers
# Uses ML to suggest best crops based on location, climate, and soil
# Version 2.0

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, Tuple, List
from dataclasses import dataclass


# Indian States and their typical characteristics
INDIAN_STATES_DATA = {
    'Punjab': {'region': 'North', 'climate': 'Temperate', 'rainfall_avg': 650},
    'Haryana': {'region': 'North', 'climate': 'Temperate', 'rainfall_avg': 600},
    'Uttar Pradesh': {'region': 'North', 'climate': 'Temperate', 'rainfall_avg': 800},
    'Madhya Pradesh': {'region': 'Central', 'climate': 'Semi-arid', 'rainfall_avg': 1100},
    'Rajasthan': {'region': 'West', 'climate': 'Arid', 'rainfall_avg': 550},
    'Gujarat': {'region': 'West', 'climate': 'Semi-arid', 'rainfall_avg': 800},
    'Maharashtra': {'region': 'West', 'climate': 'Semi-arid', 'rainfall_avg': 1200},
    'Karnataka': {'region': 'South', 'climate': 'Tropical', 'rainfall_avg': 1200},
    'Tamil Nadu': {'region': 'South', 'climate': 'Tropical', 'rainfall_avg': 1200},
    'Telangana': {'region': 'South', 'climate': 'Tropical', 'rainfall_avg': 900},
    'Andhra Pradesh': {'region': 'South', 'climate': 'Tropical', 'rainfall_avg': 900},
    'West Bengal': {'region': 'East', 'climate': 'Tropical', 'rainfall_avg': 1600},
    'Odisha': {'region': 'East', 'climate': 'Tropical', 'rainfall_avg': 1600},
    'Jharkhand': {'region': 'East', 'climate': 'Tropical', 'rainfall_avg': 1400},
    'Bihar': {'region': 'East', 'climate': 'Temperate', 'rainfall_avg': 1200},
    'Assam': {'region': 'Northeast', 'climate': 'Tropical', 'rainfall_avg': 2200},
    'Meghalaya': {'region': 'Northeast', 'climate': 'Tropical', 'rainfall_avg': 2500},
    'Himachal Pradesh': {'region': 'North', 'climate': 'Alpine', 'rainfall_avg': 1600},
    'Uttarakhand': {'region': 'North', 'climate': 'Temperate', 'rainfall_avg': 1500},
    'Kerala': {'region': 'South', 'climate': 'Tropical', 'rainfall_avg': 2800},
}

# Farming Income Data - Annual income potential per acre in INR
FARMING_INCOME_DATA = {
    'Wheat': {'min_income': 15000, 'avg_income': 25000, 'max_income': 35000, 'season': 'Rabi (Oct-Mar)'},
    'Rice': {'min_income': 20000, 'avg_income': 35000, 'max_income': 50000, 'season': 'Kharif (Jun-Sep)'},
    'Sugarcane': {'min_income': 40000, 'avg_income': 60000, 'max_income': 80000, 'season': 'Year-round'},
    'Maize': {'min_income': 18000, 'avg_income': 30000, 'max_income': 42000, 'season': 'Kharif/Rabi'},
    'Cotton': {'min_income': 22000, 'avg_income': 38000, 'max_income': 55000, 'season': 'Kharif (Jun-Oct)'},
    'Soybean': {'min_income': 16000, 'avg_income': 26000, 'max_income': 36000, 'season': 'Kharif (Jun-Sep)'},
    'Groundnut': {'min_income': 15000, 'avg_income': 25000, 'max_income': 35000, 'season': 'Kharif (Jun-Oct)'},
    'Jowar': {'min_income': 12000, 'avg_income': 20000, 'max_income': 28000, 'season': 'Kharif (Jun-Oct)'},
    'Bajra': {'min_income': 10000, 'avg_income': 18000, 'max_income': 26000, 'season': 'Kharif (Jun-Oct)'},
    'Barley': {'min_income': 14000, 'avg_income': 23000, 'max_income': 32000, 'season': 'Rabi (Oct-Mar)'},
    'Coconut': {'min_income': 50000, 'avg_income': 75000, 'max_income': 100000, 'season': 'Year-round'},
    'Tea': {'min_income': 60000, 'avg_income': 90000, 'max_income': 120000, 'season': 'Year-round'},
    'Turmeric': {'min_income': 35000, 'avg_income': 55000, 'max_income': 75000, 'season': 'Monsoon (Jun-Jul)'},
    'Ginger': {'min_income': 40000, 'avg_income': 65000, 'max_income': 90000, 'season': 'Monsoon (Jun-Jul)'},
    'Pulse': {'min_income': 14000, 'avg_income': 24000, 'max_income': 34000, 'season': 'Kharif/Rabi'},
    'Mustard': {'min_income': 16000, 'avg_income': 26000, 'max_income': 36000, 'season': 'Rabi (Oct-Mar)'},
    'Peanut': {'min_income': 14000, 'avg_income': 24000, 'max_income': 34000, 'season': 'Kharif (Jun-Oct)'},
    'Paddy': {'min_income': 22000, 'avg_income': 40000, 'max_income': 60000, 'season': 'Kharif (Jun-Sep)'},
}


@dataclass
class CropRecommendation:
    # Stores crop recommendation results
    crop_name: str
    confidence: float
    suitability_score: float
    profitability_index: float
    climate_match: float
    soil_compatibility: float


class CropEngine:
    # Main engine that loads the model and makes crop recommendations
    
    def __init__(self):
        # Load the ML model and encoders
        try:
            self.model = joblib.load('crop_model.pkl')
            self.encoders = joblib.load('encoders.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.feature_names = joblib.load('feature_names.pkl')
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
            print("⚠ Warning: Model artifacts not found. Please run model_build.py first.")
    
    def get_region_from_state(self, state: str) -> str:
        # Get the region for the given state
        return INDIAN_STATES_DATA.get(state, {}).get('region', 'Central')
    
    def calculate_climate_suitability(self, temp: float, humidity: float, 
                                     rainfall: float, crop_name: str) -> float:
        # Calculate climate suitability score for the crop (0-100)
        crop_requirements = {
            'Wheat': {'temp': (15, 25), 'humidity': (40, 70), 'rainfall': (400, 1000)},
            'Rice': {'temp': (20, 30), 'humidity': (70, 90), 'rainfall': (1200, 2500)},
            'Sugarcane': {'temp': (21, 27), 'humidity': (60, 85), 'rainfall': (1250, 2250)},
            'Maize': {'temp': (18, 27), 'humidity': (50, 80), 'rainfall': (500, 1500)},
            'Cotton': {'temp': (18, 30), 'humidity': (50, 75), 'rainfall': (600, 1000)},
            'Soybean': {'temp': (20, 30), 'humidity': (55, 80), 'rainfall': (450, 800)},
            'Groundnut': {'temp': (20, 30), 'humidity': (40, 70), 'rainfall': (500, 1000)},
            'Jowar': {'temp': (21, 30), 'humidity': (30, 70), 'rainfall': (400, 900)},
            'Bajra': {'temp': (20, 30), 'humidity': (30, 60), 'rainfall': (300, 700)},
            'Barley': {'temp': (10, 20), 'humidity': (35, 65), 'rainfall': (300, 800)},
            'Coconut': {'temp': (24, 28), 'humidity': (70, 90), 'rainfall': (1500, 2500)},
            'Tea': {'temp': (13, 24), 'humidity': (70, 90), 'rainfall': (1500, 3000)},
            'Turmeric': {'temp': (20, 30), 'humidity': (75, 90), 'rainfall': (1750, 2250)},
            'Ginger': {'temp': (22, 30), 'humidity': (80, 95), 'rainfall': (1500, 2250)},
            'Pulse': {'temp': (15, 28), 'humidity': (50, 75), 'rainfall': (500, 1200)},
            'Mustard': {'temp': (10, 20), 'humidity': (40, 70), 'rainfall': (300, 900)},
            'Peanut': {'temp': (20, 28), 'humidity': (50, 75), 'rainfall': (500, 1000)},
            'Paddy': {'temp': (20, 30), 'humidity': (75, 85), 'rainfall': (1200, 2500)},
        }
        
        if crop_name not in crop_requirements:
            return 50.0
        
        reqs = crop_requirements[crop_name]
        scores = []
        
        temp_min, temp_max = reqs['temp']
        temp_optimal = (temp_min + temp_max) / 2
        temp_deviation = abs(temp - temp_optimal)
        temp_range = temp_max - temp_min
        temp_score = max(0, 100 * (1 - (temp_deviation / (temp_range * 2))))
        scores.append(temp_score)
        
        hum_min, hum_max = reqs['humidity']
        hum_optimal = (hum_min + hum_max) / 2
        hum_deviation = abs(humidity - hum_optimal)
        hum_range = hum_max - hum_min
        hum_score = max(0, 100 * (1 - (hum_deviation / (hum_range * 2))))
        scores.append(hum_score)
        
        rain_min, rain_max = reqs['rainfall']
        rain_optimal = (rain_min + rain_max) / 2
        rain_deviation = abs(rainfall - rain_optimal)
        rain_range = rain_max - rain_min
        rain_score = max(0, 100 * (1 - (rain_deviation / (rain_range * 2))))
        scores.append(rain_score)
        
        return np.mean(scores)
    
    def calculate_soil_compatibility(self, soil_type: str, pH: float, 
                                    crop_name: str) -> float:
        # Calculate soil compatibility score for the crop (0-100)
        crop_soil_prefs = {
            'Wheat': {'preferred': ['Loamy'], 'pH': (6.0, 7.5)},
            'Rice': {'preferred': ['Loamy', 'Clayey'], 'pH': (5.5, 7.0)},
            'Sugarcane': {'preferred': ['Loamy', 'Clayey'], 'pH': (6.0, 8.0)},
            'Maize': {'preferred': ['Loamy'], 'pH': (5.8, 7.0)},
            'Cotton': {'preferred': ['Loamy', 'Sandy'], 'pH': (6.0, 8.0)},
            'Soybean': {'preferred': ['Loamy'], 'pH': (6.0, 7.5)},
            'Groundnut': {'preferred': ['Sandy', 'Loamy'], 'pH': (5.9, 7.0)},
            'Jowar': {'preferred': ['Loamy', 'Sandy'], 'pH': (6.0, 8.0)},
            'Bajra': {'preferred': ['Sandy', 'Loamy'], 'pH': (6.0, 8.5)},
            'Barley': {'preferred': ['Loamy'], 'pH': (6.0, 7.5)},
            'Coconut': {'preferred': ['Sandy', 'Loamy'], 'pH': (5.5, 8.0)},
            'Tea': {'preferred': ['Loamy'], 'pH': (4.5, 5.5)},
            'Turmeric': {'preferred': ['Loamy'], 'pH': (5.5, 7.5)},
            'Ginger': {'preferred': ['Loamy'], 'pH': (5.5, 7.0)},
            'Pulse': {'preferred': ['Loamy'], 'pH': (6.0, 7.5)},
            'Mustard': {'preferred': ['Loamy'], 'pH': (6.0, 7.5)},
            'Peanut': {'preferred': ['Sandy', 'Loamy'], 'pH': (5.9, 7.0)},
            'Paddy': {'preferred': ['Clayey', 'Loamy'], 'pH': (5.5, 7.0)},
        }
        
        if crop_name not in crop_soil_prefs:
            return 50.0
        
        prefs = crop_soil_prefs[crop_name]
        soil_score = 100.0 if soil_type in prefs['preferred'] else 50.0
        
        pH_min, pH_max = prefs['pH']
        pH_optimal = (pH_min + pH_max) / 2
        pH_deviation = abs(pH - pH_optimal)
        pH_range = pH_max - pH_min
        pH_score = max(0, 100 * (1 - (pH_deviation / (pH_range * 2))))
        
        return (soil_score * 0.4 + pH_score * 0.6)
    
    def calculate_profitability_index(self, crop_name: str, 
                                     target_income_level: str) -> float:
        # Calculate profitability index based on crop and income level (0-100)
        crop_income_mapping = {
            'Very_High': ['Rice', 'Sugarcane', 'Turmeric', 'Coconut', 'Tea', 'Paddy', 'Ginger'],
            'High': ['Wheat', 'Maize', 'Cotton', 'Soybean', 'Barley', 'Mustard'],
            'Medium': ['Jowar', 'Bajra', 'Pulse', 'Groundnut', 'Peanut'],
            'Low': []
        }
        
        income_levels = ['Low', 'Medium', 'High', 'Very_High']
        income_to_num = {'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4}
        
        actual_income = 'Medium'
        for level, crops in crop_income_mapping.items():
            if crop_name in crops:
                actual_income = level
                break
        
        actual_income_num = income_to_num[actual_income]
        target_income_num = income_to_num[target_income_level]
        
        difference = abs(actual_income_num - target_income_num)
        match_score = max(0, 100 * (1 - (difference / 3)))
        
        return match_score
    
    def generate_recommendation(self, features: np.ndarray, 
                              region: str, soil_type: str, 
                              temperature: float, humidity: float,
                              rainfall: float, pH: float,
                              income_level: str) -> Dict:
        # Generate comprehensive crop recommendation
        if not self.model_loaded:
            return {'error': 'Model not loaded. Please run model_build.py first.'}
        
        crop_encoded = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        crop_name = self.encoders['crop'].inverse_transform([crop_encoded])[0]
        confidence = float(probabilities[crop_encoded]) * 100
        
        climate_score = self.calculate_climate_suitability(
            temperature, humidity, rainfall, crop_name
        )
        soil_score = self.calculate_soil_compatibility(
            soil_type, pH, crop_name
        )
        profitability_score = self.calculate_profitability_index(
            crop_name, income_level
        )
        
        suitability_score = (
            confidence * 0.35 +
            climate_score * 0.35 +
            soil_score * 0.20 +
            profitability_score * 0.10
        )
        
        return {
            'crop_name': crop_name,
            'confidence': round(confidence, 2),
            'suitability_score': round(suitability_score, 2),
            'climate_match': round(climate_score, 2),
            'soil_compatibility': round(soil_score, 2),
            'profitability_index': round(profitability_score, 2)
        }


# Initialize Flask application
app = Flask(__name__)
recommendation_engine = CropEngine()

# Configuration
SOIL_TYPES = ['Loamy', 'Clayey', 'Sandy']
LAND_TYPES = ['Flat', 'Hilly', 'Undulating', 'Steep']
INCOME_LEVELS = ['Low', 'Medium', 'High', 'Very_High']
INDIAN_STATES = sorted(list(INDIAN_STATES_DATA.keys()))

CROP_DATABASE = {
    'Wheat': {'description': 'Winter crop, requires cool climate, high market demand', 'season': 'Winter (Oct-Mar)', 'avg_income': 'Medium'},
    'Rice': {'description': 'Water-intensive, requires warm climate, staple food', 'season': 'Monsoon-Winter (Jun-Mar)', 'avg_income': 'Very High'},
    'Sugarcane': {'description': 'Long duration crop, high water requirement, high income', 'season': 'Year-round (12 months)', 'avg_income': 'Very High'},
    'Maize': {'description': 'Versatile crop, moderate water needs, multiple uses', 'season': 'Kharif/Summer (Apr-Oct)', 'avg_income': 'High'},
    'Cotton': {'description': 'Cash crop, requires warm climate, high market value', 'season': 'Kharif (Jun-Oct)', 'avg_income': 'High'},
    'Soybean': {'description': 'Oil crop, moderate water needs, nitrogen-fixing', 'season': 'Kharif (Jun-Sep)', 'avg_income': 'Medium'},
    'Groundnut': {'description': 'Oil crop, drought-resistant, short duration', 'season': 'Kharif (Jun-Oct)', 'avg_income': 'Medium'},
    'Jowar': {'description': 'Drought-resistant, forage and grain crop', 'season': 'Kharif (Jun-Oct)', 'avg_income': 'Low-Medium'},
    'Bajra': {'description': 'Millet, drought-resistant, low input crop', 'season': 'Kharif (Jun-Oct)', 'avg_income': 'Low'},
    'Barley': {'description': 'Winter crop, cold-hardy, quick-maturing', 'season': 'Winter (Oct-Mar)', 'avg_income': 'Medium'},
    'Coconut': {'description': 'Perennial crop, high income, multiple products', 'season': 'Year-round', 'avg_income': 'High'},
    'Tea': {'description': 'Perennial crop, high-value, hillside cultivation', 'season': 'Year-round', 'avg_income': 'High'},
    'Turmeric': {'description': 'Spice crop, high market value, medium water needs', 'season': 'Monsoon (Jun-Jul)', 'avg_income': 'Very High'},
    'Ginger': {'description': 'Spice crop, shade-loving, high-value', 'season': 'Monsoon (Jun-Jul)', 'avg_income': 'High'},
    'Pulse': {'description': 'Legume crop, protein-rich, nitrogen-fixing', 'season': 'Kharif/Rabi (Jun-Oct/Oct-Mar)', 'avg_income': 'Medium'},
    'Mustard': {'description': 'Oil crop, winter season, quick-maturing', 'season': 'Winter (Oct-Mar)', 'avg_income': 'Medium'},
    'Peanut': {'description': 'Oil crop, drought-tolerant, good for sandy soils', 'season': 'Kharif (Jun-Oct)', 'avg_income': 'Medium'},
    'Paddy': {'description': 'Main staple, water-intensive, high yield', 'season': 'Monsoon-Winter (Jun-Mar)', 'avg_income': 'Very High'},
}


@app.route('/')
def index():
    # Return the home page
    return render_template('index.html',
                         states=INDIAN_STATES,
                         soil_types=SOIL_TYPES,
                         land_types=LAND_TYPES,
                         income_levels=INCOME_LEVELS)


@app.route('/api/recommend', methods=['POST'])
def recommend_crop():
    # API endpoint to recommend crops based on input data
    if not recommendation_engine.model_loaded:
        return jsonify({'error': 'Model not loaded. Please run model_build.py first.'}), 500
    
    try:
        data = request.json
        
        required_fields = ['state', 'city', 'temperature', 'humidity', 'rainfall',
                          'soil_type', 'land_type', 'pH', 'income_level']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Extract input
        state = str(data['state']).strip()
        city = str(data['city']).strip()
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        rainfall = float(data['rainfall'])
        soil_type = str(data['soil_type']).strip()
        land_type = str(data['land_type']).strip()
        pH = float(data['pH'])
        income_level = str(data['income_level']).strip()
        
        # Get region from state
        region = recommendation_engine.get_region_from_state(state)
        
        # Encode categorical variables with error handling
        try:
            region_encoded = recommendation_engine.encoders['region'].transform([region])[0]
        except Exception as e:
            return jsonify({'error': f'Region encoding error: {str(e)}. Region value: {region}'}), 400
            
        try:
            soil_encoded = recommendation_engine.encoders['soil_type'].transform([soil_type])[0]
        except Exception as e:
            return jsonify({'error': f'Soil type encoding error: {str(e)}. Value: {soil_type}'}), 400
            
        try:
            land_encoded = recommendation_engine.encoders['land_type'].transform([land_type])[0]
        except Exception as e:
            return jsonify({'error': f'Land type encoding error: {str(e)}. Value: {land_type}'}), 400
            
        try:
            income_encoded = recommendation_engine.encoders['income_level'].transform([income_level])[0]
        except Exception as e:
            return jsonify({'error': f'Income level encoding error: {str(e)}. Value: {income_level}'}), 400
        
        # Prepare feature array
        features = np.array([[
            region_encoded,
            temperature,
            humidity,
            rainfall,
            soil_encoded,
            land_encoded,
            pH,
            income_encoded
        ]])
        
        # Generate recommendation
        rec_result = recommendation_engine.generate_recommendation(
            features, region, soil_type, temperature, humidity,
            rainfall, pH, income_level
        )
        
        crop_name = rec_result['crop_name']
        crop_details = CROP_DATABASE.get(crop_name, {
            'description': 'No information available',
            'season': 'Check local guidelines',
            'avg_income': 'Variable'
        })
        
        # Get farming income data
        income_data = FARMING_INCOME_DATA.get(crop_name, {
            'min_income': 0,
            'avg_income': 0,
            'max_income': 0,
            'season': 'Check local guidelines'
        })
        
        return jsonify({
            'success': True,
            'state': state,
            'city': city,
            'recommended_crop': crop_name,
            'confidence': rec_result['confidence'],
            'suitability_score': rec_result['suitability_score'],
            'climate_match': rec_result['climate_match'],
            'soil_compatibility': rec_result['soil_compatibility'],
            'profitability_index': rec_result['profitability_index'],
            'description': crop_details['description'],
            'season': income_data['season'],
            'avg_income': crop_details['avg_income'],
            'income_details': {
                'min_income_per_acre': f"₹{income_data['min_income']:,}",
                'avg_income_per_acre': f"₹{income_data['avg_income']:,}",
                'max_income_per_acre': f"₹{income_data['max_income']:,}",
                'annual_income_1acre': f"₹{income_data['avg_income']:,}",
                'annual_income_5acre': f"₹{income_data['avg_income'] * 5:,}",
                'annual_income_10acre': f"₹{income_data['avg_income'] * 10:,}",
            },
            'summary': f"Recommended crop for {city}, {state}: {crop_name} with {rec_result['suitability_score']}% "
                      f"overall suitability. Expected annual income: ₹{income_data['avg_income']:,} per acre."
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/api/crops-info')
def crops_info():
    # Return crop database
    return jsonify(CROP_DATABASE)


@app.route('/api/farming-income')
def farming_income():
    # Return farming income data
    return jsonify(FARMING_INCOME_DATA)


@app.route('/api/states')
def states_list():
    # Return list of states
    return jsonify(list(INDIAN_STATES_DATA.keys()))


if __name__ == '__main__':
    if not recommendation_engine.model_loaded:
        print("\n❌ ERROR: Model artifacts not found.")
        print("   Please run: python model_build.py")
    else:
        print("\n✅ Recommendation engine initialized successfully!")
        print("   Starting Flask development server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
