
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


df = pd.read_csv(r"C:\Users\Nithya\Desktop\ml3\diabetes.csv")

print("Dataset shape:", df.shape)
print(df.head())


X = df.drop("Outcome", axis=1)


for col in ["Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI"]:
    X[col] = X[col].replace(0, X[col].median())


X = StandardScaler().fit_transform(X)


kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X)

print("\nK-Means Silhouette Score:",
      silhouette_score(X, k_labels))


gmm = GaussianMixture(n_components=2, random_state=42)
g_labels = gmm.fit_predict(X)

print("EM Silhouette Score:",
      silhouette_score(X, g_labels))


print("\nK-Means Results:")
print(pd.crosstab(k_labels, df["Outcome"]))

print("\nEM Results:")
print(pd.crosstab(g_labels, df["Outcome"]))
