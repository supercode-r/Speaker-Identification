import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class SpeakerDataAnalyzer:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.features_df = None
        self.label_counts = None
        
    def load_data(self, train_file='train.txt', test_file='test.txt', labels_file='train_block_labels.txt'):
        """Load and parse the training and test data"""
        print("Loading data...")
        
        # Load training data
        with open(train_file, 'r') as f:
            content = f.read().strip()
        
        # Split into blocks
        blocks = content.split('\n\n')
        self.train_data = []
        
        for block in blocks:
            if block.strip():  # Skip empty blocks
                lines = block.strip().split('\n')
                block_data = []
                for line in lines:
                    values = [float(x) for x in line.split()]
                    block_data.append(values)
                self.train_data.append(np.array(block_data))
        
        # Load test data
        with open(test_file, 'r') as f:
            content = f.read().strip()
        
        blocks = content.split('\n\n')
        self.test_data = []
        
        for block in blocks:
            if block.strip():
                lines = block.strip().split('\n')
                block_data = []
                for line in lines:
                    values = [float(x) for x in line.split()]
                    block_data.append(values)
                self.test_data.append(np.array(block_data))
        
        # Load labels
        with open(labels_file, 'r') as f:
            label_counts = [int(x) for x in f.read().strip().split()]
        
        self.label_counts = label_counts
        self.train_labels = []
        
        # Create labels for each block
        for label, count in enumerate(label_counts):
            self.train_labels.extend([label] * count)
        
        print(f"Loaded {len(self.train_data)} training blocks")
        print(f"Loaded {len(self.test_data)} test blocks")
        print(f"Label distribution: {dict(enumerate(label_counts))}")
        
    def analyze_data_structure(self):
        """Analyze the basic structure of the data"""
        print("\n" + "="*50)
        print("DATA STRUCTURE ANALYSIS")
        print("="*50)
        
        # Analyze sequence lengths
        train_lengths = [len(block) for block in self.train_data]
        test_lengths = [len(block) for block in self.test_data]
        
        print(f"Training sequence lengths: {min(train_lengths)} to {max(train_lengths)}")
        print(f"Test sequence lengths: {min(test_lengths)} to {max(test_lengths)}")
        print(f"Mean training length: {np.mean(train_lengths):.2f}")
        print(f"Mean test length: {np.mean(test_lengths):.2f}")
        
        # Analyze feature dimensions
        print(f"Feature dimensions: {self.train_data[0].shape[1]}")
        
        # Class distribution analysis
        print("\nClass Distribution:")
        for i, count in enumerate(self.label_counts):
            percentage = (count / sum(self.label_counts)) * 100
            print(f"Speaker {i}: {count} utterances ({percentage:.1f}%)")
        
        # Check for class imbalance
        max_count = max(self.label_counts)
        min_count = min(self.label_counts)
        imbalance_ratio = max_count / min_count
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        
        return train_lengths, test_lengths
    
    def visualize_data_patterns(self):
        """Create visualizations to understand data patterns"""
        print("\n" + "="*50)
        print("DATA VISUALIZATION")
        print("="*50)
        
        # Set up the plotting
        plt.figure(figsize=(20, 15))
        
        # 1. Sequence length distribution
        plt.subplot(3, 4, 1)
        train_lengths = [len(block) for block in self.train_data]
        plt.hist(train_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Sequence Lengths')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        
        # 2. Class distribution
        plt.subplot(3, 4, 2)
        speakers = list(range(len(self.label_counts)))
        plt.bar(speakers, self.label_counts, color='lightcoral', edgecolor='black')
        plt.title('Speaker Distribution')
        plt.xlabel('Speaker ID')
        plt.ylabel('Number of Utterances')
        
        # 3. Feature statistics by speaker
        plt.subplot(3, 4, 3)
        speaker_means = []
        for speaker in range(len(self.label_counts)):
            speaker_blocks = [self.train_data[i] for i, label in enumerate(self.train_labels) if label == speaker]
            speaker_mean = np.mean([np.mean(block) for block in speaker_blocks])
            speaker_means.append(speaker_mean)
        
        plt.bar(speakers, speaker_means, color='lightgreen', edgecolor='black')
        plt.xticks(ticks=range(len(self.label_counts)), labels=[f'S{i}' for i in range(len(self.label_counts))])
        plt.title('Average Feature Values by Speaker')
        plt.xlabel('Speaker')
        plt.ylabel('Average Value')
        plt.xticks(rotation=45)

        
        # 4. First coefficient over time for sample utterances
        plt.subplot(3, 4, 4)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.label_counts)))
        for speaker in range(len(self.label_counts)):  # Show all speakers
            speaker_indices = [i for i, label in enumerate(self.train_labels) if label == speaker]
            if speaker_indices:
                sample_block = self.train_data[speaker_indices[0]]
                plt.plot(sample_block[:, 0], color=colors[speaker], label=f'Speaker {speaker}', alpha=0.7)
        
        plt.title('First LPC Coefficient Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Coefficient Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5-8. Distribution of each coefficient across speakers
        for coef_idx in range(4):
            plt.subplot(3, 4, 5 + coef_idx)
            coef_data = []
            labels = []
            
            for speaker in range(len(self.label_counts)):
                speaker_blocks = [self.train_data[i] for i, label in enumerate(self.train_labels) if label == speaker]
                speaker_coef_values = []
                for block in speaker_blocks:
                    speaker_coef_values.extend(block[:, coef_idx])
                coef_data.append(speaker_coef_values)
                labels.append(f'S{speaker}')
            
            plt.boxplot(coef_data, labels=labels)
            plt.title(f'Coefficient {coef_idx + 1} Distribution')
            plt.xticks(rotation=45)
            plt.ylabel('Coefficient Value')
        
        # 9. Sequence length by speaker
        plt.subplot(3, 4, 9)
        length_by_speaker = []
        for speaker in range(len(self.label_counts)):
            speaker_lengths = [len(self.train_data[i]) for i, label in enumerate(self.train_labels) if label == speaker]
            length_by_speaker.append(speaker_lengths)
        
        plt.boxplot(length_by_speaker, labels=[f'S{i}' for i in range(len(self.label_counts))])
        plt.title('Sequence Length by Speaker')
        plt.xlabel('Speaker')
        plt.ylabel('Sequence Length')
        plt.xticks(rotation=45)
        
        # 10. Correlation heatmap of first block features
        plt.subplot(3, 4, 10)
        sample_block = self.train_data[0]
        correlation = np.corrcoef(sample_block.T)
        sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation')
        
        # 11. Feature variance across time
        plt.subplot(3, 4, 11)
        variances = []
        for i in range(12):  # 12 LPC coefficients
            coef_variances = []
            for block in self.train_data:
                coef_variances.append(np.var(block[:, i]))
            variances.append(np.mean(coef_variances))
        
        plt.bar(range(12), variances, color='gold', edgecolor='black')
        plt.title('Average Variance per Coefficient')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Average Variance')
        
        # 12. PCA visualization
        plt.subplot(3, 4, 12)
        features = self.extract_basic_features()
        if features is not None:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.label_counts)))
            for speaker in range(len(self.label_counts)):
                mask = np.array(self.train_labels) == speaker
                plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                           c=[colors[speaker]], label=f'Speaker {speaker}', alpha=0.6)
            
            plt.title('PCA Visualization')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def extract_basic_features(self):
        """Extract basic statistical features from the time series"""
        print("\n" + "="*50)
        print("FEATURE EXTRACTION")
        print("="*50)
        
        features = []
        feature_names = []
        
        # Generate feature names
        coef_names = [f'coef_{i}' for i in range(12)]
        stat_names = ['mean', 'std', 'min', 'max', 'median', 'ptp']
        
        for coef in coef_names:
            for stat in stat_names:
                feature_names.append(f'{coef}_{stat}')

        # Add temporal feature names
        for coef in coef_names:
            feature_names.extend([
                f'{coef}_first',
                f'{coef}_last',
                f'{coef}_delta',
                f'{coef}_slope',
                f'{coef}_curvature'
            ])
        
        # Add global block-level feature names
        feature_names.extend([
            'seq_length',
            'avg_energy',
            'energy_std',
            'corr_coef_0_1',
            'corr_coef_0_2',
            'corr_coef_1_2'
        ])

        for block in self.train_data:
            block_features = []

            for coeff_idx in range(12):
                coeff_data = block[:, coeff_idx]

                # Statistical features
                block_features.extend([
                    np.mean(coeff_data),
                    np.std(coeff_data),
                    np.median(coeff_data),
                    np.min(coeff_data),
                    np.max(coeff_data),
                    np.ptp(coeff_data)
                ])

                # Temporal features
                if len(coeff_data) > 1:
                    first = coeff_data[0]
                    last = coeff_data[-1]
                    delta = last - first

                    # Linear slope
                    x = np.arange(len(coeff_data))
                    slope = np.polyfit(x, coeff_data, 1)[0]

                    # Curvature (2nd derivative approximation)
                    if len(coeff_data) > 2:
                        curvature = np.mean(np.diff(coeff_data, n=2))
                    else:
                        curvature = 0
                else:
                    first = last = coeff_data[0]
                    delta = slope = curvature = 0

                block_features.extend([first, last, delta, slope, curvature])

            # Global features
            seq_length = len(block)
            energy = np.sum(block**2, axis=1)
            avg_energy = np.mean(energy)
            energy_std = np.std(energy)

            # Cross-coefficient correlations (only if enough time steps)
            if len(block) > 1:
                corr_01 = np.corrcoef(block[:, 0], block[:, 1])[0, 1]
                corr_02 = np.corrcoef(block[:, 0], block[:, 2])[0, 1]
                corr_12 = np.corrcoef(block[:, 1], block[:, 2])[0, 1]
            else:
                corr_01 = corr_02 = corr_12 = 0

            block_features.extend([
                seq_length,
                avg_energy,
                energy_std,
                corr_01,
                corr_02,
                corr_12
            ])
            
            features.append(block_features)
        
        features = np.array(features)
        
        # Create DataFrame
        self.features_df = pd.DataFrame(features, columns=feature_names)
        self.features_df['speaker'] = self.train_labels
        
        print(f"Extracted {features.shape[1]} features from {features.shape[0]} blocks")
        print(f"Feature matrix shape: {features.shape}")
        
        return features
    
    def _safe_skew(self, data):
        """Safely compute skewness"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _safe_kurtosis(self, data):
        """Safely compute kurtosis"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def analyze_feature_importance(self):
        """Analyze which features are most important for speaker identification"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        if self.features_df is None:
            print("Please run extract_basic_features() first")
            return
        
        # Prepare data
        X = self.features_df.drop('speaker', axis=1)
        y = self.features_df['speaker']
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 20 Most Important Features:")
        print(importance_df.head(20))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def quick_model_evaluation(self):
        """Quick model evaluation to understand baseline performance"""
        print("\n" + "="*50)
        print("QUICK MODEL EVALUATION")
        print("="*50)
        
        if self.features_df is None:
            print("Please run extract_basic_features() first")
            return
        
        # Prepare data
        X = self.features_df.drop('speaker', axis=1)
        y = self.features_df['speaker']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        # Train on full dataset and show per-class performance
        rf.fit(X_scaled, y)
        y_pred = rf.predict(X_scaled)
        
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Show confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'Speaker {i}' for i in range(len(self.label_counts))],
                    yticklabels=[f'Speaker {i}' for i in range(len(self.label_counts))])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Per-speaker accuracy (key metric for this problem)
        per_speaker_accuracy = []
        for speaker in range(len(self.label_counts)):
            speaker_mask = (y == speaker)
            if np.sum(speaker_mask) > 0:
                accuracy = np.mean(y_pred[speaker_mask] == y[speaker_mask])
                per_speaker_accuracy.append(accuracy)
                print(f"Speaker {speaker} accuracy: {accuracy:.4f}")
        
        worst_speaker_accuracy = min(per_speaker_accuracy)
        print(f"\nWorst speaker accuracy: {worst_speaker_accuracy:.4f}")
        print(f"This is your key metric for the competition!")
        
        return rf, scaler, per_speaker_accuracy
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE DATA ANALYSIS SUMMARY")
        print("="*60)
        
        # Data overview
        print("1. DATA OVERVIEW:")
        print(f"   - Training blocks: {len(self.train_data)}")
        print(f"   - Test blocks: {len(self.test_data)}")
        print(f"   - Number of speakers: {len(self.label_counts)}")
        print(f"   - Feature dimensions: 12 LPC coefficients")
        
        # Class imbalance
        print("\n2. CLASS IMBALANCE ANALYSIS:")
        max_count = max(self.label_counts)
        min_count = min(self.label_counts)
        print(f"   - Most samples: {max_count} (Speaker {self.label_counts.index(max_count)})")
        print(f"   - Least samples: {min_count} (Speaker {self.label_counts.index(min_count)})")
        print(f"   - Imbalance ratio: {max_count/min_count:.2f}")
        print("   - Recommendation: Use stratified sampling and class weights")
        
        # Sequence length analysis
        train_lengths = [len(block) for block in self.train_data]
        print(f"\n3. SEQUENCE LENGTH ANALYSIS:")
        print(f"   - Range: {min(train_lengths)} to {max(train_lengths)} time steps")
        print(f"   - Mean: {np.mean(train_lengths):.2f}")
        print(f"   - Recommendation: Sequence length could be a discriminative feature")
        
        print("\n4. RECOMMENDATIONS:")
        print("   - Focus on robust features that work across all speakers")
        print("   - Use stratified validation to ensure all speakers are represented")
        print("   - Consider ensemble methods to improve worst-case performance")
        print("   - Pay special attention to speakers with fewer samples")
        print("   - Temporal features (slopes, first/last values) seem promising")

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SpeakerDataAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Run comprehensive analysis
    analyzer.analyze_data_structure()
    analyzer.visualize_data_patterns()
    
    # Extract features
    analyzer.extract_basic_features()
    
    # Analyze feature importance
    analyzer.analyze_feature_importance()
    
    # Quick model evaluation
    analyzer.quick_model_evaluation()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)