import numpy as np
import pandas as pd
import joblib
from scipy.signal import hilbert
from PyEMD import EMD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PowerSystemEventTester:
    def __init__(self, model_path, scaler_path):
        """Initialize with pre-trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.emd = EMD()
        print("Pre-trained model and scaler loaded successfully!")
    
    def generate_ieee30_test_data(self, n_samples=300):
        """
        Generate test data for IEEE 30 bus system
        Different characteristics from IEEE 69 bus to test generalization
        """
        np.random.seed(456)  # Different seed for testing
        
        # IEEE 30 bus system parameters
        base_voltage = 1.0
        base_frequency = 60.0  # Different frequency (60Hz vs 50Hz in training)
        
        data = []
        labels = []
        
        for i in range(n_samples):
            # Random event type with different distribution
            event_type = np.random.choice([0, 1, 2], p=[0.55, 0.25, 0.20])
            
            # IEEE 30 bus system
            n_buses = 30
            n_time_points = 100
            
            if event_type == 0:  # Normal operation
                voltage_mag = base_voltage + np.random.normal(0, 0.018, (n_buses, n_time_points))
                frequency = base_frequency + np.random.normal(0, 0.09, n_time_points)
                
            elif event_type == 1:  # Generator outage
                voltage_mag = base_voltage + np.random.normal(0, 0.018, (n_buses, n_time_points))
                # Generator buses in IEEE 30 system (buses 1, 2, 5, 8, 11, 13)
                generator_buses = [0, 1, 4, 7, 10, 12]  # 0-indexed
                outage_bus = np.random.choice(generator_buses)
                
                # Voltage drop due to generator outage
                voltage_mag[outage_bus, 50:] *= (0.82 + np.random.uniform(0, 0.09))
                
                # Cascading effect on nearby buses based on electrical distance
                for j in range(n_buses):
                    if j != outage_bus:
                        # Create electrical distance effect
                        electrical_distance = abs(j - outage_bus)
                        if electrical_distance <= 5:
                            distance_factor = 1.0 - 0.03 * electrical_distance
                            voltage_mag[j, 50:] *= (distance_factor + np.random.uniform(-0.02, 0.02))
                
                frequency = base_frequency + np.random.normal(0, 0.18, n_time_points)
                frequency[50:] += np.random.uniform(-0.7, -0.15)  # Frequency drop
                
            else:  # Line trip (event_type == 2)
                voltage_mag = base_voltage + np.random.normal(0, 0.018, (n_buses, n_time_points))
                
                # Critical transmission lines in IEEE 30-bus system
                critical_lines = [
                    (0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (1, 5), 
                    (3, 5), (4, 6), (5, 7), (5, 8), (8, 9), (8, 27),
                    (5, 9), (9, 10), (11, 12), (11, 13), (12, 14), 
                    (12, 15), (12, 16), (14, 17), (15, 18), (17, 19),
                    (18, 19), (19, 20), (19, 21), (21, 22), (22, 23),
                    (23, 24), (24, 25), (25, 26), (25, 27), (27, 28), (28, 29)
                ]
                bus1, bus2 = critical_lines[np.random.randint(len(critical_lines))]
                
                # Voltage oscillations after line trip
                oscillation_freq = 0.12 + np.random.uniform(0, 0.08)
                damping_factor = 0.04 + np.random.uniform(0, 0.02)
                
                time_points_after_trip = n_time_points - 45
                damping = np.exp(-damping_factor * np.arange(time_points_after_trip))
                
                voltage_mag[bus1, 45:] += 0.12 * np.sin(2 * np.pi * np.arange(time_points_after_trip) * oscillation_freq) * damping
                voltage_mag[bus2, 45:] += 0.12 * np.cos(2 * np.pi * np.arange(time_points_after_trip) * oscillation_freq) * damping
                
                # Secondary oscillations on adjacent buses
                adjacent_buses = [max(0, bus1-1), min(n_buses-1, bus1+1), max(0, bus2-1), min(n_buses-1, bus2+1)]
                for adj_bus in adjacent_buses:
                    if adj_bus != bus1 and adj_bus != bus2:
                        voltage_mag[adj_bus, 45:] += 0.06 * np.sin(2 * np.pi * np.arange(time_points_after_trip) * oscillation_freq * 1.2) * damping
                
                frequency = base_frequency + np.random.normal(0, 0.14, n_time_points)
                frequency[45:] += 0.18 * np.sin(2 * np.pi * np.arange(time_points_after_trip) * 0.07) * damping
            
            # Adapt to training format (use first 10 buses to match training)
            adapted_voltage_mag = voltage_mag[:10, :]
            
            sample_data = {
                'voltage_mag': adapted_voltage_mag,
                'frequency': frequency
            }
            
            data.append(sample_data)
            labels.append(event_type)
            
        return data, labels
    
    def extract_emd_features(self, signal):
        """Extract EMD features (same as training)"""
        try:
            imfs = self.emd(signal)
            
            features = []
            for i in range(min(3, len(imfs))):
                imf = imfs[i]
                features.extend([
                    np.mean(imf),
                    np.std(imf),
                    np.max(imf),
                    np.min(imf),
                    np.sum(np.abs(imf))
                ])
            
            while len(features) < 15:
                features.append(0.0)
                
            return features[:15]
        except:
            return [0.0] * 15
    
    def extract_svd_features(self, signal_matrix):
        """Extract SVD features (same as training)"""
        try:
            U, s, Vt = np.linalg.svd(signal_matrix, full_matrices=False)
            
            features = [
                np.sum(s),
                np.max(s),
                np.min(s),
                np.std(s),
                s[0] / np.sum(s) if np.sum(s) > 0 else 0,
                len(s[s > 0.01 * np.max(s)])
            ]
            return features
        except:
            return [0.0] * 6
    
    def extract_hilbert_features(self, signal):
        """Extract Hilbert Transform features (same as training)"""
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
        """Extract all features from a data sample (same as training)"""
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
    
    def prepare_test_data(self, data):
        """Prepare feature matrix for testing"""
        feature_matrix = []
        
        print("Extracting features from IEEE 30 bus test data...")
        for i, sample in enumerate(data):
            if i % 50 == 0:
                print(f"Processing sample {i}/{len(data)}")
            
            features = self.extract_all_features(sample)
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
    def test_model(self, test_data, test_labels):
        """Test the model on IEEE 30 bus data"""
        print("Testing model on IEEE 30 bus system...")
        
        # Extract features
        X_test = self.prepare_test_data(test_data)
        
        # Scale features using training scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        return y_pred, y_pred_proba
    
    def evaluate_results(self, y_true, y_pred, y_pred_proba):
        """Evaluate and display test results"""
        print("\nTesting Results on IEEE 30 Bus System")
        print("=" * 45)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {accuracy:.3f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                    target_names=['Normal', 'Generator Outage', 'Line Trip']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Event-wise analysis
        print("\nEvent-wise Performance:")
        for i, event_name in enumerate(['Normal', 'Generator Outage', 'Line Trip']):
            true_events = np.sum(y_true == i)
            detected_events = np.sum((y_true == i) & (y_pred == i))
            if true_events > 0:
                detection_rate = detected_events / true_events
                print(f"{event_name}: {detected_events}/{true_events} detected ({detection_rate:.3f})")
        
        # Generate all visualization plots
        self.plot_confusion_matrix(cm)
        self.plot_prediction_confidence(y_true, y_pred_proba)
        self.plot_performance_metrics(y_true, y_pred)
        self.plot_event_distribution(y_true, y_pred)
        
        return accuracy, cm
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8},
                    xticklabels=['Normal', 'Generator Outage', 'Line Trip'],
                    yticklabels=['Normal', 'Generator Outage', 'Line Trip'])
        plt.title('Confusion Matrix - IEEE 30 Bus System Test Results', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_confidence(self, y_true, y_pred_proba):
        """Plot prediction confidence distribution"""
        plt.figure(figsize=(15, 5))
        
        event_names = ['Normal', 'Generator Outage', 'Line Trip']
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for i in range(3):
            plt.subplot(1, 3, i+1)
            
            # Get confidence scores for this class
            class_indices = y_true == i
            if np.sum(class_indices) > 0:
                class_confidences = y_pred_proba[class_indices, i]
                
                plt.hist(class_confidences, bins=15, alpha=0.7, edgecolor='black', 
                        color=colors[i], label=f'{event_names[i]}')
                plt.title(f'{event_names[i]}\nConfidence Distribution', fontweight='bold')
                plt.xlabel('Prediction Confidence')
                plt.ylabel('Count')
                plt.xlim(0, 1)
                plt.grid(True, alpha=0.3)
                
                # Add statistics
                mean_conf = np.mean(class_confidences)
                plt.axvline(mean_conf, color='red', linestyle='--', 
                          label=f'Mean: {mean_conf:.3f}')
                plt.legend()
        
        plt.suptitle('Prediction Confidence Analysis - IEEE 30 Bus System', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, y_true, y_pred):
        """Plot performance metrics by class"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        event_names = ['Normal', 'Generator Outage', 'Line Trip']
        x_pos = np.arange(len(event_names))
        
        # Precision, Recall, F1-Score
        width = 0.25
        ax1.bar(x_pos - width, precision, width, label='Precision', alpha=0.8, color='lightblue')
        ax1.bar(x_pos, recall, width, label='Recall', alpha=0.8, color='lightcoral')
        ax1.bar(x_pos + width, fscore, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        ax1.set_xlabel('Event Type')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics by Event Type')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(event_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Support (number of samples)
        bars = ax2.bar(event_names, support, color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax2.set_xlabel('Event Type')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Test Sample Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_event_distribution(self, y_true, y_pred):
        """Plot event distribution comparison"""
        plt.figure(figsize=(12, 6))
        
        event_names = ['Normal', 'Generator Outage', 'Line Trip']
        
        # Count actual vs predicted
        true_counts = [np.sum(y_true == i) for i in range(3)]
        pred_counts = [np.sum(y_pred == i) for i in range(3)]
        
        x_pos = np.arange(len(event_names))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, true_counts, width, label='Actual', 
                       alpha=0.8, color='steelblue')
        bars2 = plt.bar(x_pos + width/2, pred_counts, width, label='Predicted', 
                       alpha=0.8, color='orange')
        
        plt.xlabel('Event Type')
        plt.ylabel('Number of Events')
        plt.title('Event Distribution: Actual vs Predicted (IEEE 30 Bus System)')
        plt.xticks(x_pos, event_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def real_time_detection_demo(self, n_samples=15):
        """Demonstrate real-time event detection"""
        print("\nReal-time Event Detection Demo on IEEE 30 Bus System")
        print("=" * 55)
        
        # Generate demo samples
        demo_data, demo_labels = self.generate_ieee30_test_data(n_samples)
        
        event_names = ['Normal Operation', 'Generator Outage', 'Line Trip']
        
        correct_predictions = 0
        
        for i in range(n_samples):
            # Extract features
            features = self.extract_all_features(demo_data[i])
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0]
            
            # Check if prediction is correct
            is_correct = "✓" if prediction == demo_labels[i] else "✗"
            if prediction == demo_labels[i]:
                correct_predictions += 1
            
            print(f"Sample {i+1:2d}: True={event_names[demo_labels[i]]:<18} | "
                  f"Predicted={event_names[prediction]:<18} | "
                  f"Confidence={confidence[prediction]:.3f} | {is_correct}")
        
        demo_accuracy = correct_predictions / n_samples
        print(f"\nDemo Accuracy: {correct_predictions}/{n_samples} ({demo_accuracy:.3f})")

def main():
    """Main testing function"""
    print("Power System Event Detection - Testing Phase (IEEE 30 Bus)")
    print("=" * 60)
    
    # Load pre-trained model (same names as in training file)
    model_path = "power_event_model_classifier.pkl"
    scaler_path = "power_event_model_scaler.pkl"
    
    try:
        # Initialize tester with pre-trained model
        tester = PowerSystemEventTester(model_path, scaler_path)
        
        # Generate IEEE 30 bus test data
        print("Generating IEEE 30 bus system test data...")
        test_data, test_labels = tester.generate_ieee30_test_data(n_samples=300)
        
        print(f"Generated {len(test_data)} test samples")
        print(f"Normal operations: {test_labels.count(0)}")
        print(f"Generator outages: {test_labels.count(1)}")
        print(f"Line trips: {test_labels.count(2)}")
        
        # Test the model
        y_pred, y_pred_proba = tester.test_model(test_data, test_labels)
        
        # Evaluate results with comprehensive visualization
        accuracy, cm = tester.evaluate_results(test_labels, y_pred, y_pred_proba)
        
        # Real-time detection demo
        tester.real_time_detection_demo()
        
        print(f"\nTesting completed successfully!")
        print(f"Final accuracy on IEEE 30 bus system: {accuracy:.3f}")
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"Total test samples: {len(test_labels)}")
        print(f"Correctly classified: {np.sum(y_pred == test_labels)}")
        print(f"Misclassified: {np.sum(y_pred != test_labels)}")
        
    except FileNotFoundError as e:
        print("Error: Model files not found!")
        print("Please run the training script first to generate the model files.")
        print("Expected files:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        print("Please check that the model files are compatible and properly saved.")

if __name__ == "__main__":
    main()