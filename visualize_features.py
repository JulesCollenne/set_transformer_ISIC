import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":
    models = ("BYOL", "Moco", "SimCLR", "normalized/SwaV", "normalized/SimSiam")
    for model in models:
        print(model)
        df = pd.read_csv(f"features/{model}_test.csv")

        feature_columns = [col for col in df.columns if col.startswith('features')]
        features = df[feature_columns]
        targets = df['target']

        # Perform dimensionality reduction with t-SNE
        tsne = TSNE(n_components=2, random_state=42)  # You can adjust the number of components as needed (e.g., n_components=2 for 2D visualization)
        tsne_result = tsne.fit_transform(features)

        # Create a new dataframe for the t-SNE results
        tsne_df = pd.DataFrame(data=tsne_result, columns=['Dimension 1', 'Dimension 2'])

        # Add the 'target' column to the t-SNE dataframe
        tsne_df['target'] = targets

        # Visualize the t-SNE results with different colors based on the 'target' column
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_df[tsne_df['target'] == 0]['Dimension 1'], tsne_df[tsne_df['target'] == 0]['Dimension 2'], c='blue', label='Target 0', marker='o', s=10)
        plt.scatter(tsne_df[tsne_df['target'] == 1]['Dimension 1'], tsne_df[tsne_df['target'] == 1]['Dimension 2'], c='red', label='Target 1', marker='o', s=10)
        plt.title('t-SNE Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()

        plt.show()
