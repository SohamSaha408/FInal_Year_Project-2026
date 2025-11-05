import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import sys

# Define a minimal required crop map for visualization and processing consistency
CROP_MAP = {'Rice': 0, 'Wheat': 1, 'Maize': 2}

class NDVIYieldPredictor:
    
    def __init__(self):
        # Model and components
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Features required in the final DataFrame for training
        self.feature_cols = ['NDVI', 'EVI', 'LST', 'Rainfall', 'Soil_Moisture', 'Season', 'Crop_Type']
        self.crop_map = CROP_MAP

    def load_vedas_ndvi_data(self, csv_path):
        """Loads the raw time-series NDVI data from the CSV file."""
        try:
            # We assume no header in the first column (DateTime) based on the raw data structure
            data = pd.read_csv(csv_path) 
            # Data loaded successfully (suppress verbose output)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at '{csv_path}'. Check your file path.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error loading VEDAS data: {e}", file=sys.stderr)
            return None

    # --- NEW: SYNTHESIS FUNCTION (REPLACES REAL DATA) ---
    def _synthesize_data_from_ndvi(self, ndvi, month, year):
        """
        Synthesizes ALL required features (LST, Rainfall, Yield, etc.) from a single NDVI value.
        ***NOTE: This MUST be replaced by real data merging in a production system.***
        """
        
        # Season and Crop Type Logic
        season_map = {
            'Jun': 0, 'Jul': 0, 'Aug': 0, 'Sep': 0, 'Oct': 0,  # Kharif
            'Nov': 1, 'Dec': 1, 'Jan': 1, 'Feb': 1, 'Mar': 1, 'Apr': 1, 'May': 0 
        }
        season = season_map.get(month, 0)
        
        # ESTIMATE OTHER PARAMETERS
        evi = ndvi * 0.8  
        lst = (35 + (ndvi * 10)) if month in ['Jun', 'Jul', 'Aug'] else (25 + (ndvi * 8))
        rainfall = (800 + (ndvi * 800)) if season == 0 else (300 + (ndvi * 400))
        soil_moisture = 0.1 + (ndvi * 0.4)
        
        if ndvi > 0.6:
            crop_type = self.crop_map['Rice']  
        elif ndvi > 0.4:
            crop_type = self.crop_map['Wheat'] 
        else:
            crop_type = self.crop_map['Maize'] 
        
        # YIELD ESTIMATION (Synthetic relationship)
        base_yield = ndvi * 3500
        crop_multipliers = {0: 1.3, 1: 0.9, 2: 1.0} 
        base_yield *= crop_multipliers.get(crop_type, 1.0)
        
        yield_value = base_yield + np.random.normal(0, 200)
        yield_value = max(yield_value, 500) 
        
        return {
            'NDVI': round(ndvi, 4), 'EVI': round(evi, 4), 'LST': round(lst, 1),
            'Rainfall': round(rainfall, 0), 'Soil_Moisture': round(soil_moisture, 3),
            'Season': season, 'Crop_Type': crop_type, 'Yield': round(yield_value, 0),
            'Month': month, 'Year_Range': year # Year_Range is kept for context
        }

    # --- MODIFIED: WIDE-TO-LONG TRANSFORMATION ---
    def process_vedas_data(self, data):
        """
        Converts VEDAS wide-form time series (NDVI) into long-form,
        then synthesizes missing features and cleans the data for ML.
        """
        if data is None:
            return None
            
        # 1. WIDE to LONG (Using melt)
        year_cols = [col for col in data.columns if col not in ['DateTime']]
        
        long_data = data.melt(
            id_vars=['DateTime'], 
            value_vars=year_cols,
            var_name='Year_Range', 
            value_name='NDVI'
        )
        
        # 2. Clean up
        long_data[['Month', 'Day']] = long_data['DateTime'].str.split('-', expand=True)
        long_data['Month'] = long_data['Month'].str.capitalize()

        long_data.replace('', np.nan, inplace=True)
        long_data['NDVI'] = pd.to_numeric(long_data['NDVI'], errors='coerce')
        long_data.dropna(subset=['NDVI'], inplace=True)
        long_data = long_data[long_data['NDVI'] > 0]
        
        # 3. SYNTHESIZE MISSING FEATURES
        processed_data = []
        
        for _, row in long_data.iterrows():
            ndvi = row['NDVI']
            month = row['Month']
            year_range = row['Year_Range']
            
            features = self._synthesize_data_from_ndvi(ndvi, month, year_range)
            processed_data.append(features)
            
        final_df = pd.DataFrame(processed_data)
        
        # 4. Final Cleanup
        final_df.dropna(subset=self.feature_cols + ['Yield'], inplace=True)
        
        # Suppress verbose output during multi-file processing
        if len(final_df) > 0:
            pass  # Data processed successfully
        return final_df

    def train_model(self, data):
        """Train yield prediction model and evaluate performance."""
        
        X = data[self.feature_cols]
        y = data['Yield']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n🎯 Model Performance:")
        print(f"   R² Score: {r2:.3f}")
        print(f"   RMSE: {rmse:.0f} kg/ha")
        print(f"   MAE: {mae:.0f} kg/ha")
        
        self.plot_feature_importance()

        return y_test, y_pred
    
    def predict_yield(self, feature_vector):
        """
        Predict yield from a single feature vector (list of feature values).
        Vector must match: [NDVI, EVI, LST, Rainfall, Soil_Moisture, Season, Crop_Type]
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Please run train_model first.")
        
        X_predict = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X_predict)
        prediction = self.model.predict(X_scaled)
        
        return prediction[0]

    def plot_feature_importance(self):
        """Plots Feature Importance from the Random Forest model."""
        if not hasattr(self.model, 'feature_importances_'):
            return

        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)
        
        plt.figure(figsize=(8, 6))
        plt.title('Feature Importance (Random Forest)')
        plt.barh(np.array(self.feature_cols)[sorted_indices], importances[sorted_indices], color='darkred')
        plt.xlabel('Relative Importance Score')
        plt.tight_layout()
        plt.show()

    def analyze_ndvi_yield_relationship(self, data, y_test=None, y_pred=None):
        """Analyzes and visualizes the data relationships and model residuals."""
        
        plt.figure(figsize=(18, 12))
        
        # ... (Visualization code remains the same as previously modified)
        
        # 1. NDVI vs Yield scatter
        plt.subplot(2, 3, 1)
        plt.scatter(data['NDVI'], data['Yield'], alpha=0.6, c=data['Season'], cmap='viridis')
        plt.xlabel('NDVI')
        plt.ylabel('Yield (kg/ha)')
        plt.title('1. NDVI vs Yield by Season')
        plt.colorbar(label='Season (0=Kharif, 1=Rabi)')
        
        # 2. Correlation Heatmap
        plt.subplot(2, 3, 2)
        import seaborn as sns
        corr = data[self.feature_cols + ['Yield']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar=True)
        plt.title('2. Feature Correlation Heatmap')
        
        # 3. Box Plot: Yield Distribution by Crop Type
        plt.subplot(2, 3, 3)
        crop_names = {v: k for k, v in self.crop_map.items()}
        data['Crop_Name'] = data['Crop_Type'].map(crop_names)
        data.boxplot(column='Yield', by='Crop_Name', grid=False, color='goldenrod')
        plt.title('3. Yield Volatility by Crop Type (Box Plot)')
        plt.suptitle('') 
        plt.xlabel('Crop Type')
        plt.ylabel('Yield (kg/ha)')
        
        # 4. Residuals Plot
        if y_test is not None and y_pred is not None:
            plt.subplot(2, 3, 4)
            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6, color='darkblue')
            plt.hlines(0, y_pred.min(), y_pred.max(), color='red', linestyle='--')
            plt.title('4. Residuals vs. Predicted Yield')
            plt.xlabel('Predicted Yield (kg/ha)')
            plt.ylabel('Residuals (Actual - Predicted)')
            plt.grid(True, linestyle=':', alpha=0.6)
        
        # 5. Actual vs Predicted Scatter
        if y_test is not None and y_pred is not None:
            plt.subplot(2, 3, 5)
            plt.scatter(y_test, y_pred, alpha=0.7, color='purple')
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            plt.title('5. Actual vs. Predicted Yield')
            plt.xlabel('Actual Yield (kg/ha)')
            plt.ylabel('Predicted Yield (kg/ha)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def save_model(self, filepath):
        """Save trained model and scaler"""
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_cols}, filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.feature_cols = loaded['features']
        self.is_trained = True
        print(f"Model loaded: {filepath}")
    
    def predict_single_yield(self, input_file, show_graph=True):
        """Give one decisive yield prediction for a single input file"""
        print(f"\n{'='*60}")
        print(f"🎯 SINGLE YIELD PREDICTION")
        print(f"{'='*60}")
        print(f"📄 Input File: {input_file}")
        
        # Load and process data
        vedas_data = self.load_vedas_ndvi_data(input_file)
        if vedas_data is None:
            print("❌ File not found or invalid format")
            return None
            
        processed_data = self.process_vedas_data(vedas_data.copy())
        if processed_data is None or len(processed_data) < 5:
            print("❌ Insufficient data for prediction")
            return None
        
        # Train model on this data
        print(f"🤖 Training model on {len(processed_data)} data points...")
        y_test, y_pred = self.train_model(processed_data)
        
        # Calculate average NDVI and predict yield
        avg_ndvi = processed_data['NDVI'].mean()
        avg_conditions = [
            avg_ndvi,
            avg_ndvi * 0.8,  # EVI
            processed_data['LST'].mean(),
            processed_data['Rainfall'].mean(),
            processed_data['Soil_Moisture'].mean(),
            processed_data['Season'].mode()[0],
            processed_data['Crop_Type'].mode()[0]
        ]
        
        predicted_yield = self.predict_yield(avg_conditions)
        
        # Display decisive result
        print(f"\n{'='*60}")
        print(f"🎯 DECISIVE YIELD PREDICTION")
        print(f"{'='*60}")
        print(f"🌾 Expected Yield: {predicted_yield:.0f} kg/ha")
        print(f"📈 Average NDVI: {avg_ndvi:.3f}")
        
        # Yield category
        if predicted_yield > 3000:
            category = "🏆 EXCELLENT YIELD (>3000 kg/ha)"
        elif predicted_yield > 2000:
            category = "🥇 GOOD YIELD (2000-3000 kg/ha)"
        elif predicted_yield > 1000:
            category = "🥈 AVERAGE YIELD (1000-2000 kg/ha)"
        else:
            category = "⚠️ LOW YIELD (<1000 kg/ha)"
        
        print(f"🏷️ Category: {category}")
        
        # Show confidence based on model performance
        from sklearn.metrics import r2_score
        confidence = r2_score(y_test, y_pred) * 100
        print(f"🎯 Confidence: {confidence:.1f}%")
        
        if show_graph:
            self.plot_single_prediction_graph(processed_data, predicted_yield, avg_ndvi)
        
        return {
            'predicted_yield': predicted_yield,
            'avg_ndvi': avg_ndvi,
            'confidence': confidence,
            'category': category
        }
    
    def plot_single_prediction_graph(self, data, predicted_yield, avg_ndvi):
        """Plot decisive prediction graph"""
        plt.figure(figsize=(12, 8))
        
        # Main prediction plot
        plt.subplot(2, 2, 1)
        plt.scatter(data['NDVI'], data['Yield'], alpha=0.6, color='lightblue', s=50)
        plt.axvline(avg_ndvi, color='red', linestyle='--', linewidth=2, label=f'Your NDVI: {avg_ndvi:.3f}')
        plt.axhline(predicted_yield, color='green', linestyle='-', linewidth=3, label=f'Predicted: {predicted_yield:.0f} kg/ha')
        plt.xlabel('NDVI')
        plt.ylabel('Yield (kg/ha)')
        plt.title('🎯 Your Yield Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # NDVI trend
        plt.subplot(2, 2, 2)
        plt.plot(range(len(data)), data['NDVI'], marker='o', color='green', linewidth=2)
        plt.axhline(avg_ndvi, color='red', linestyle='--', label=f'Average: {avg_ndvi:.3f}')
        plt.xlabel('Time Period')
        plt.ylabel('NDVI')
        plt.title('📈 NDVI Trend Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Yield distribution
        plt.subplot(2, 2, 3)
        plt.hist(data['Yield'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(predicted_yield, color='red', linestyle='-', linewidth=3, label=f'Your Prediction: {predicted_yield:.0f}')
        plt.xlabel('Yield (kg/ha)')
        plt.ylabel('Frequency')
        plt.title('📊 Yield Distribution')
        plt.legend()
        
        # Summary box
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = f"""
        🎯 PREDICTION SUMMARY
        
        📄 File: {len(data)} data points
        🌱 Avg NDVI: {avg_ndvi:.3f}
        🌾 Predicted Yield: {predicted_yield:.0f} kg/ha
        
        📊 Data Range:
        • Min Yield: {data['Yield'].min():.0f} kg/ha
        • Max Yield: {data['Yield'].max():.0f} kg/ha
        • Avg Yield: {data['Yield'].mean():.0f} kg/ha
        """
        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def train_multiple_files(self, file_list, save_individual_models=False):
        """Train model on multiple CSV files and combine data"""
        print(f"\n{'='*60}")
        print(f"🚀 MULTI-FILE TRAINING: {len(file_list)} FILES")
        print(f"{'='*60}")
        
        all_processed_data = []
        successful_files = []
        failed_files = []
        
        # Process each file with progress bar
        for i, filename in enumerate(file_list):
            progress = f"[{i+1:2d}/{len(file_list)}]"
            print(f"\n{progress} Processing: {filename}...", end=" ")
            
            try:
                # Load data (suppress verbose output)
                vedas_data = pd.read_csv(filename) if filename else None
                
                if vedas_data is not None:
                    # Process data (suppress verbose output)
                    processed_data = self.process_vedas_data(vedas_data.copy())
                    
                    if processed_data is not None and len(processed_data) >= 5:
                        processed_data['Source_File'] = filename
                        all_processed_data.append(processed_data)
                        successful_files.append(filename)
                        print(f"✅ ({len(processed_data)} points)")
                    else:
                        failed_files.append((filename, "Insufficient data"))
                        print("❌ No data")
                else:
                    failed_files.append((filename, "File not found"))
                    print("❌ Not found")
                    
            except Exception as e:
                failed_files.append((filename, str(e)[:30]))
                print(f"❌ Error")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(successful_files)}/{len(file_list)} files")
        print(f"❌ Failed: {len(failed_files)} files")
        
        if failed_files:
            print(f"\n⚠️  Failed Files:")
            for fname, reason in failed_files[:5]:  # Show first 5 failures
                print(f"   • {fname}: {reason}")
            if len(failed_files) > 5:
                print(f"   ... and {len(failed_files)-5} more")
        
        # Combine all data
        if all_processed_data:
            combined_data = pd.concat(all_processed_data, ignore_index=True)
            print(f"\n🔗 Combined Dataset: {combined_data.shape[0]:,} samples from {len(successful_files)} files")
            
            # Train on combined data
            print(f"\n{'='*60}")
            print(f"🤖 TRAINING COMBINED MODEL")
            print(f"{'='*60}")
            y_test, y_pred = self.train_model(combined_data)
            
            # Calculate final accuracy
            from sklearn.metrics import r2_score, mean_squared_error
            final_r2 = r2_score(y_test, y_pred)
            final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"\n{'='*60}")
            print(f"🎯 FINAL MODEL PERFORMANCE")
            print(f"{'='*60}")
            print(f"📈 Accuracy (R²): {final_r2:.1%}")
            print(f"📉 Error (RMSE): {final_rmse:.0f} kg/ha")
            
            if final_r2 > 0.9:
                print(f"🏆 EXCELLENT MODEL! (>90% accuracy)")
            elif final_r2 > 0.8:
                print(f"🥇 VERY GOOD MODEL! (>80% accuracy)")
            elif final_r2 > 0.7:
                print(f"🥈 GOOD MODEL (>70% accuracy)")
            else:
                print(f"⚠️  Model needs improvement (<70% accuracy)")
            
            # Analyze combined results
            self.analyze_ndvi_yield_relationship(combined_data, y_test, y_pred)
            
            # Save combined model
            self.save_model('combined_model.pkl')
            print(f"\n💾 Model saved as: combined_model.pkl")
            
            return combined_data, successful_files
        else:
            print("\n❌ NO FILES PROCESSED SUCCESSFULLY")
            print("   Check file paths and formats")
            return None, []


# Main execution
if __name__ == "__main__":
    predictor = NDVIYieldPredictor()
    
    print("🛰️ ISRO VEDAS NDVI Yield Prediction System")
    print("=" * 50)
    
    # Option 1: Train on single file
    single_file = 'Nashik_grape_field.csv'
    
    # Option 2: Train on multiple files
    multiple_files = [
      'Sent_(1).csv',
      'Sent_(2).csv',
      'Sent_(3).csv',
      'Sent_(4).csv',
      'Sent_(5).csv',
      'Sent_(6).csv',
      'Sent_(7).csv',
      'Sent_(8).csv',
      'Sent_(9).csv',
      'Sent_(10).csv',
      'Sent_(11).csv',
      'Sent_(12).csv',
      'Sent_(13).csv',
      'Sent_(14).csv',
      'Sent_(15).csv',
      'Sent_(16).csv',
      'Sent_(17).csv',
      'Sent_(18).csv',
      'Sent_(19).csv',
      'Sent_(20).csv',
      'Sent_(21).csv',
      'Sent_(22).csv',
      'Sent_(23).csv',
      'Sent_(24).csv',
      'Sent_(25).csv',
      'Sent_(26).csv',
      'Sent_(27).csv',
      'Sent_(28).csv',
      'Sent_(29).csv',
      'Sent_(30).csv',
      'Sent_(31).csv',
      'Sent_(32).csv',
      'Sent_(33).csv',
      'Sent_(34).csv',
      'Sent_(35).csv',
      'Sent_(36).csv',
      'Sent_(37).csv',
      
      'Sent_(38).csv'
      ]
    
    # Choose prediction mode
    prediction_mode = "single"  # Options: "single", "multiple", "train_single"
    
    if prediction_mode == "single":
        print("\n🎯 SINGLE FILE DECISIVE PREDICTION")
        result = predictor.predict_single_yield(single_file)
        
    elif prediction_mode == "multiple":
        print("\n📁 MULTIPLE FILE TRAINING MODE")
        combined_data, successful_files = predictor.train_multiple_files(multiple_files)
        
        if combined_data is not None:
            print(f"\n✅ Training completed on {len(successful_files)} files")
            
            # Example prediction
            example_features = [0.8, 0.64, 30.5, 950.0, 0.35, 0, 0]
            try:
                pred_yield = predictor.predict_yield(example_features)
                print(f"\n🔮 Example Prediction (Combined Model): {pred_yield:.0f} kg/ha")
            except ValueError as e:
                print(f"Error during prediction: {e}")
    
    else:  # train_single mode
        print("\n📄 SINGLE FILE TRAINING MODE")
        vedas_data = predictor.load_vedas_ndvi_data(single_file)
        
        if vedas_data is not None:
            processed_data = predictor.process_vedas_data(vedas_data.copy())
            
            if processed_data is not None and len(processed_data) >= 10:
                print("\n🤖 Training yield prediction model...")
                y_test, y_pred = predictor.train_model(processed_data)
                
                predictor.analyze_ndvi_yield_relationship(processed_data, y_test, y_pred)
                predictor.save_model('single_file_model.pkl')
                
                example_features = [0.8, 0.64, 30.5, 950.0, 0.35, 0, 0]
                try:
                    pred_yield = predictor.predict_yield(example_features)
                    print(f"\n🔮 Example Prediction (Single File): {pred_yield:.0f} kg/ha")
                except ValueError as e:
                    print(f"Error during prediction: {e}")
            else:
                print("❌ Insufficient valid data points after processing (need at least 10).")
        else:
            print("❌ System halted due to data loading/processing error.")
