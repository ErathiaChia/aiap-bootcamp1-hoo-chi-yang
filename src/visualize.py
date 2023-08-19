
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_model_performance(test_scores):
        model_names = list(test_scores.keys())
        mse_values = list(test_scores.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(model_names, mse_values, color='skyblue')
        plt.xlabel('Model Name')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
