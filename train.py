import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

def train_model():
    # 1. Sample Data (from your notebook)
    data = {
        "Weight": [45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        "Height": [150, 155, 160, 162, 165, 170, 172, 175, 178, 180]
    }
    df = pd.DataFrame(data)

    # 2. Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # 3. Train Agglomerative Clustering Model
    # We choose 3 clusters as a baseline for this capstone
    model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    model.fit(scaled_data)

    # 4. Save the objects
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model and Scaler saved successfully!")

if __name__ == "__main__":
    train_model()