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
    
    def generate_ieee57_test_data(self, n_samples=200):
        """
        Generate test data for IEEE 57 bus system
        Different characteristics from IEEE 69 bus to test generalization
        """
        np.random.seed(123)  # Different seed for testing
        
        # IEEE 57 bus system parameters
        base_voltage = 1.0
        base_frequency = 60.0  # Different frequency (60Hz vs 50Hz in training)
        
        data = []
        labels = []
        
        for i in range(n_samples):
            # Random event type with different distribution
            event_type = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            
            # IEEE 57 bus system
            n_buses = 57
            n_time_points = 100
            
            if event_type == 0:  # Normal operation
                voltage_mag = base_voltage + np.random.normal(0, 0.018, (n_buses, n_time_points))
                frequency = base_frequency + np.random.normal(0, 0.09, n_time_points)
                
            elif event_type == 1:  # Generator outage
                voltage_mag = base_voltage + np.random.normal(0, 0.018, (n_buses, n_time_points))
                # Generator buses in IEEE 57 system (buses 1, 2, 3, 6, 8, 9, 12 are typical generator buses)
                generator_buses = [0, 1, 2, 5, 7, 8, 11]  # 0-indexed
                outage_bus = np.random.choice(generator_buses)
                
                # Voltage drop due to generator outage
                voltage_mag[outage_bus, 52:] *= (0.82 + np.random.uniform(0, 0.10))
                
                # Cascading effect on nearby buses with network topology consideration
                affected_buses = []
                for j in range(n_buses):
                    if j != outage_bus:
                        # Distance-based effect with some randomness for network topology
                        distance = min(abs(j - outage_bus), 8)
                        if distance <= 3:
                            affected_buses.append(j)
                            impact_factor = 0.95 - 0.03 * distance + np.random.uniform(-0.01, 0.01)
                            voltage_mag[j, 52:] *= impact_factor
                
                frequency = base_frequency + np.random.normal(0, 0.18, n_time_points)
                frequency[52:] += np.random.uniform(-0.7, -0.15)  # Frequency drop
                
            else:  # Line trip (event_type == 2)
                voltage_mag = base_voltage + np.random.normal(0, 0.018, (n_buses, n_time_points))
                
                # Critical transmission lines in IEEE 57-bus system
                # These represent important interconnections
                critical_lines = [
                    (0, 1), (1, 2), (2, 3), (1, 16), (2, 4), (3, 5),
                    (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
                    (10, 12), (11, 13), (12, 14), (13, 15), (24, 25),
                    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30)
                ]
                bus1, bus2 = critical_lines[np.random.randint(len(critical_lines))]
                
                # Voltage oscillations after line trip
                oscillation_freq = 0.12 + np.random.uniform(0, 0.08)
                phase_shift = np.random.uniform(0, 2*np.pi)
                
                # Primary affected buses
                voltage_mag[bus1, 42:] += 0.12 * np.sin(2 * np.pi * np.arange(n_time_points-42) * oscillation_freq + phase_shift)
                voltage_mag[bus2, 42:] += 0.12 * np.cos(2 * np.pi * np.arange(n_time_points-42) * oscillation_freq + phase_shift)
                
                # Secondary effects on nearby buses
                for nearby_bus in range(max(0, min(bus1, bus2) - 3), min(n_buses, max(bus1, bus2) + 4)):
                    if nearby_bus not in [bus1, bus2]:
                        voltage_mag[nearby_bus, 42:] += 0.06 * np.sin(2 * np.pi * np.arange(n_time_points-42) * oscillation_freq * 0.8 + phase_shift)
                
                # Damped oscillations
                damping = np.exp(-0.04 * np.arange(n_time_points-42))
                voltage_mag[bus1, 42:] *= damping
                voltage_mag[bus2, 42:] *= damping
                
                frequency = base_frequency + np.random.normal(0, 0.14, n_time_points)
                frequency[42:] += 0.18 * np.sin(2 * np.pi * np.arange(n_time_points-42) * 0.07) * damping
            
            # Use first 10 buses for compatibility with training model
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
        
        print("Extracting features from IEEE 57 bus test data...")
        for i, sample in enumerate(data):
            if i % 50 == 0:
                print(f"Processing sample {i}/{len(data)}")
            
            features = self.extract_all_features(sample)
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
    def test_model(self, test_data, test_labels):
        """Test the model on IEEE 57 bus data"""
        print("Testing model on IEEE 57 bus system...")
        
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
        print("\nTesting Results on IEEE 57 Bus System")
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
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Plot prediction confidence
        self.plot_prediction_confidence(y_true, y_pred_proba)
        
        # Plot performance metrics
        self.plot_performance_metrics(y_true, y_pred)
        
        # Plot feature importance
        self.plot_feature_importance()
        
        return accuracy, cm
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                    xticklabels=['Normal', 'Generator Outage', 'Line Trip'],
                    yticklabels=['Normal', 'Generator Outage', 'Line Trip'])
        plt.title('Confusion Matrix - IEEE 57 Bus Test Results', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
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
                
                plt.hist(class_confidences, bins=20, alpha=0.7, color=colors[i], 
                        edgecolor='black', linewidth=1.2)
                plt.title(f'{event_names[i]}\nConfidence Distribution', fontsize=12, fontweight='bold')
                plt.xlabel('Prediction Confidence', fontsize=11)
                plt.ylabel('Count', fontsize=11)
                plt.xlim(0, 1)
                plt.grid(True, alpha=0.3)
                
                # Add statistics
                mean_conf = np.mean(class_confidences)
                plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_conf:.3f}')
                plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, y_true, y_pred):
        """Plot performance metrics visualization"""
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate metrics for each class
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        
        event_names = ['Normal', 'Generator Outage', 'Line Trip']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Precision, Recall, F1-Score
        x = np.arange(len(event_names))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', color='skyblue', alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', color='lightcoral', alpha=0.8)
        ax1.bar(x + width, f1, width, label='F1-Score', color='lightgreen', alpha=0.8)
        
        ax1.set_xlabel('Event Types', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Performance Metrics by Event Type', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(event_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Support (number of samples)
        ax2.bar(event_names, support, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax2.set_xlabel('Event Types', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Test Sample Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy by class
        class_accuracies = []
        for i in range(3):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_true == i) & (y_pred == i)) / np.sum(class_mask)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        ax3.bar(event_names, class_accuracies, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax3.set_xlabel('Event Types', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Class-wise Accuracy', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Overall metrics summary
        overall_metrics = ['Accuracy', 'Macro Avg Precision', 'Macro Avg Recall', 'Macro Avg F1']
        overall_values = [
            accuracy_score(y_true, y_pred),
            np.mean(precision),
            np.mean(recall),
            np.mean(f1)
        ]
        
        ax4.bar(overall_metrics, overall_values, color='gold', alpha=0.8)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Overall Performance Summary', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance from the trained model"""
        if hasattr(self.model, 'feature_importances_'):
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create feature names (simplified)
            feature_names = []
            for bus in range(10):
                feature_names.extend([f'Bus{bus+1}_EMD{i+1}' for i in range(15)])
                feature_names.extend([f'Bus{bus+1}_Hilbert{i+1}' for i in range(6)])
            feature_names.extend([f'SVD_Feature{i+1}' for i in range(6)])
            feature_names.extend([f'Freq_EMD{i+1}' for i in range(15)])
            feature_names.extend([f'Freq_Hilbert{i+1}' for i in range(6)])
            
            # Get top 20 most important features
            top_indices = np.argsort(importances)[-20:]
            top_importances = importances[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_importances)), top_importances, color='steelblue', alpha=0.8)
            plt.yticks(range(len(top_importances)), top_names, fontsize=10)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title('Top 20 Feature Importances - IEEE 57 Bus System', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def real_time_detection_demo(self, n_samples=10):
        """Demonstrate real-time event detection"""
        print("\nReal-time Event Detection Demo")
        print("=" * 35)
        
        # Generate a few test samples
        demo_data, demo_labels = self.generate_ieee57_test_data(n_samples)
        
        event_names = ['Normal Operation', 'Generator Outage', 'Line Trip']
        
        print(f"{'Sample':<8} {'True Event':<18} {'Predicted Event':<18} {'Confidence':<12} {'Status'}")
        print("-" * 70)
        
        correct_predictions = 0
        for i in range(n_samples):
            # Extract features
            features = self.extract_all_features(demo_data[i])
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0]
            
            # Check if prediction is correct
            status = "✓" if prediction == demo_labels[i] else "✗"
            if prediction == demo_labels[i]:
                correct_predictions += 1
            
            print(f"{i+1:>6}   {event_names[demo_labels[i]]:<18} {event_names[prediction]:<18} "
                  f"{confidence[prediction]:.3f}        {status}")
        
        print("-" * 70)
        print(f"Demo Accuracy: {correct_predictions}/{n_samples} ({correct_predictions/n_samples:.1%})")

def main():
    """Main testing function"""
    print("Power System Event Detection - Testing Phase")
    print("=" * 50)
    
    # Load pre-trained model (adjust paths as needed)
    model_path = "power_event_model_classifier.pkl"
    scaler_path = "power_event_model_scaler.pkl"
    
    try:
        # Initialize tester with pre-trained model
        tester = PowerSystemEventTester(model_path, scaler_path)
        
        # Generate IEEE 57 bus test data
        print("Generating IEEE 57 bus system test data...")
        test_data, test_labels = tester.generate_ieee57_test_data(n_samples=200)
        
        print(f"Generated {len(test_data)} test samples")
        print(f"Normal operations: {test_labels.count(0)}")
        print(f"Generator outages: {test_labels.count(1)}")
        print(f"Line trips: {test_labels.count(2)}")
        
        # Test the model
        y_pred, y_pred_proba = tester.test_model(test_data, test_labels)
        
        # Evaluate results
        accuracy, cm = tester.evaluate_results(test_labels, y_pred, y_pred_proba)
        
        # Real-time detection demo
        tester.real_time_detection_demo()
        
        print(f"\nTesting completed successfully!")
        print(f"Final accuracy on IEEE 57 bus system: {accuracy:.3f}")
        
    except FileNotFoundError:
        print("Error: Model files not found!")
        print("Please run the training script first to generate the model files.")
        print("Expected files:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")

if __name__ == "__main__":
    main()