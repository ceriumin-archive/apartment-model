import matplotlib.pyplot as plt
import numpy as np

'''This class contains all the diagrams and graph layouts which can be executed'''
class Visualize:

    # Scatter diagram suggesting predicted values against actual values of the dataset
    def scatter(self, name, actual, predicted):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, color='blue', label='Data Points')
        plt.plot(actual, actual, color='red', linewidth=2, label='Perfect Fit')

        plt.title(f"{name}: Predicted vs Actual Values")
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Bar chart diagram suggesting the importance of specific columns in the dataset
    # Shows which columns contribute to the most accurate predictions
    def importance(self, model, name, features):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"{name}: Feature Importance")
        plt.bar(range(len(features)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
        plt.tight_layout
        plt.show()

    # Cross validation results diagram
    def validation(self, scores, name):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='blue', label='Validation Score')
        plt.axhline(y=scores.mean(), color='red', linestyle='--', label=f'Mean Accuracy')

        plt.title(f"{name}: Cross-Validation Accuracy")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, len(scores) + 1))
        plt.legend()
        plt.grid(True)
        plt.show()

    # Residual Plot diagram
    def residual(self, name, actual, predicted):
        residuals = actual - predicted

        plt.figure(figsize=(10, 6))
        plt.scatter(actual, residuals, color='blue', alpha=0.6, label='Residuals')
        plt.axhline(0, color='red', linewidth=2, linestyle='--', label='Zero Line')

        plt.title(f'{name}: Residual Plot')
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.legend()
        plt.grid(True)
        plt.show()




