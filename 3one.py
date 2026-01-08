import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from scipy.signal import hilbert
from PyEMD import EMD
import warnings
warnings.filterwarnings('ignore')

class PowerSystemEventDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.emd = EMD()
        
    def generate_ieee69_data(self, n_samples=1000):
        """
        Generate synthetic power system data for IEEE 69 bus system
        Simulates voltage magnitudes, phase angles, and frequency measurements
        """
        np.random.seed(42)
        
        # Normal operation parameters
        base_voltage = 1.0  # per unit
        base_frequency = 50.0  # Hz
        
        data = []
        labels = []
        
        for i in range(n_samples):
            # Random event type: 0=Normal, 1=Generator Outage, 2=Line Trip
            event_type = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
            
            # Time series for 69 buses (simplified representation)
            n_buses = 69
            n_time_points = 100
            
            if event_type == 0:  # Normal operation
                voltage_mag = base_voltage + np.random.normal(0, 0.02, (n_buses, n_time_points))
                frequency = base_frequency + np.random.normal(0, 0.1, n_time_points)
                
            elif event_type == 1:  # Generator outage
                voltage_mag = base_voltage + np.random.normal(0, 0.02, (n_buses, n_time_points))
                # Simulate voltage drop due to generator outage
                outage_bus = np.random.randint(0, 10)  # Generator buses typically 1-10
                voltage_mag[outage_bus, 50:] *= (0.85 + np.random.uniform(0, 0.1))
                
                # Nearby buses affected
                for j in range(max(0, outage_bus-2), min(n_buses, outage_bus+3)):
                    voltage_mag[j, 50:] *= (0.92 + np.random.uniform(0, 0.05))
                
                frequency = base_frequency + np.random.normal(0, 0.2, n_time_points)
                frequency[50:] += np.random.uniform(-0.5, -0.1)  # Frequency drop
                
            else:  # Line trip (event_type == 2)
                voltage_mag = base_voltage + np.random.normal(0, 0.02, (n_buses, n_time_points))
                # Simulate line trip between random buses
                bus1, bus2 = np.random.choice(n_buses, 2, replace=False)
                
                # Voltage oscillations after line trip
                voltage_mag[bus1, 45:] += 0.1 * np.sin(2 * np.pi * np.arange(n_time_points-45) * 0.1)
                voltage_mag[bus2, 45:] += 0.1 * np.cos(2 * np.pi * np.arange(n_time_points-45) * 0.1)
                
                frequency = base_frequency + np.random.normal(0, 0.15, n_time_points)
                frequency[45:] += 0.1 * np.sin(2 * np.pi * np.arange(n_time_points-45) * 0.05)
            
            # Store representative bus data (first 10 buses for computational efficiency)
            sample_data = {
                'voltage_mag': voltage_mag[:10, :],
                'frequency': frequency
            }
            
            data.append(sample_data)
            labels.append(event_type)
            
        return data, labels
    
    def extract_emd_features(self, signal):
        """Extract EMD (Empirical Mode Decomposition) features"""
        try:
            # Perform EMD decomposition
            imfs = self.emd(signal)
            
            features = []
            # Take first 3 IMFs for feature extraction
            for i in range(min(3, len(imfs))):
                imf = imfs[i]
                features.extend([
                    np.mean(imf),
                    np.std(imf),
                    np.max(imf),
                    np.min(imf),
                    np.sum(np.abs(imf))
                ])
            
            # Pad features if less than 3 IMFs
            while len(features) < 15:  # 3 IMFs * 5 features each
                features.append(0.0)
                
            return features[:15]  # Ensure consistent feature size
        except:
            return [0.0] * 15  # Return zero features if EMD fails
    
    def extract_svd_features(self, signal_matrix):
        """Extract SVD (Singular Value Decomposition) features"""
        try:
            U, s, Vt = np.linalg.svd(signal_matrix, full_matrices=False)
            
            features = [
                np.sum(s),  # Sum of singular values
                np.max(s),  # Maximum singular value
                np.min(s),  # Minimum singular value
                np.std(s),  # Standard deviation of singular values
                s[0] / np.sum(s) if np.sum(s) > 0 else 0,  # Dominant singular value ratio
                len(s[s > 0.01 * np.max(s)])  # Number of significant singular values
            ]
            return features
        except:
            return [0.0] * 6
    
    def extract_hilbert_features(self, signal):
        """Extract Hilbert Transform features"""
        try:
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            features = [
                np.mean(amplitude_envelope),
                np.std(amplitude_envelope),
                np.max(amplitude_envelope),
                np.mean(instantaneous_frequency),
                np.std(instantaneous_frequency),
                np.max(instantaneous_frequency) - np.min(instantaneous_frequency)
            ]
            return features
        except:
            return [0.0] * 6
    
    def extract_all_features(self, data_sample):
        """Extract all features (EMD, SVD, Hilbert) from a data sample"""
        voltage_mag = data_sample['voltage_mag']
        frequency = data_sample['frequency']
        
        all_features = []
        
        # Extract features from each bus voltage
        for bus_idx in range(voltage_mag.shape[0]):
            bus_voltage = voltage_mag[bus_idx, :]
            
            # EMD features
            emd_features = self.extract_emd_features(bus_voltage)
            all_features.extend(emd_features)
            
            # Hilbert features
            hilbert_features = self.extract_hilbert_features(bus_voltage)
            all_features.extend(hilbert_features)
        
        # SVD features from voltage magnitude matrix
        svd_features = self.extract_svd_features(voltage_mag)
        all_features.extend(svd_features)
        
        # Frequency features
        freq_emd_features = self.extract_emd_features(frequency)
        all_features.extend(freq_emd_features)
        
        freq_hilbert_features = self.extract_hilbert_features(frequency)
        all_features.extend(freq_hilbert_features)
        
        return all_features
    
    def prepare_training_data(self, data, labels):
        """Prepare feature matrix for training"""
        feature_matrix = []
        
        print("Extracting features from training data...")
        for i, sample in enumerate(data):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(data)}")
            
            features = self.extract_all_features(sample)
            feature_matrix.append(features)
        
        return np.array(feature_matrix), np.array(labels)
    
    def train_model(self, X, y):
        """Train the machine learning model"""
        print("Training the model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Validation
        y_pred = self.model.predict(X_val_scaled)
        
        print("\nTraining completed!")
        print("\nValidation Results:")
        print("==================")
        print(classification_report(y_val, y_pred, target_names=['Normal', 'Generator Outage', 'Line Trip']))
        
        return X_val_scaled, y_val, y_pred
    
    def save_model(self, filepath_base='power_event_model'):
        """Save the trained model and scaler"""
        model_path = f"{filepath_base}_classifier.pkl"
        scaler_path = f"{filepath_base}_scaler.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved as: {model_path}")
        print(f"Scaler saved as: {scaler_path}")
        
        return model_path, scaler_path

def main():
    """Main training function"""
    print("Power System Event Detection - Training Phase")
    print("=" * 50)
    
    # Initialize detector
    detector = PowerSystemEventDetector()
    
    # Generate IEEE 69 bus training data
    print("Generating IEEE 69 bus system data...")
    data, labels = detector.generate_ieee69_data(n_samples=1000)
    
    print(f"Generated {len(data)} samples")
    print(f"Normal operations: {labels.count(0)}")
    print(f"Generator outages: {labels.count(1)}")
    print(f"Line trips: {labels.count(2)}")
    
    # Prepare training data
    X, y = detector.prepare_training_data(data, labels)
    print(f"Feature matrix shape: {X.shape}")
    
    # Train model
    X_val, y_val, y_pred = detector.train_model(X, y)
    
    # Save model
    model_path, scaler_path = detector.save_model()
    
    print("\nTraining phase completed successfully!")
    print(f"Model files: {model_path}, {scaler_path}")
    
    return detector, model_path, scaler_path

if __name__ == "__main__":
    trained_detector, model_file, scaler_file = main()