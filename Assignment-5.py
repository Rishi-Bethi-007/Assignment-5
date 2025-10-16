from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd



mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:5000], X[60000:], y[:5000], y[60000:]

# === Scale your data ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def grid_searching (model_name, param_grid,X_train, X_test, y_train, y_test):
    
    start_time = time.time()  # Start timer
    
    grid_search =  GridSearchCV( model_name, param_grid,cv= 3 , n_jobs= -1, verbose= 3)
    grid_search.fit(X_train,y_train)
    
    end_time = time.time()  # End timer
    training_time = end_time - start_time
    
    best_model = grid_search.best_estimator_
    print(f'\n Best estimator is {best_model}')
    
    y_pred = grid_search.predict(X_test)
    
    print("\nBest Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(f"Training Time: {training_time:.2f} seconds")
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

    return {'Model': str(model_name),
        'Best_Params': grid_search.best_params_,
        'CV_Score': grid_search.best_score_,
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1': f1_score(y_test, y_pred, average='macro'),
        'Training_Time': training_time}
   
param_grid_linear = {
    'C': [0.1, 1, 10],          # broad range but small grid
    'loss': ['squared_hinge'],  # faster and smoother than hinge
    'dual': [False],            # faster for n_samples > n_features
    'tol': [1e-3],              # stops earlier (trades a bit of precision for speed)
    'max_iter': [2000]          # enough for convergence with scaled data # Number of iterations (helps avoid convergence warnings)
}

param_grid_rbf = {
    'C': [ 1, 10],
    'gamma': ['scale', 0.1]
}

param_grid_poly = {
    'C': [ 1, 10],
    'degree': [2, 3],
    'gamma': ['scale'],
    'coef0': [0, 1]  # Controls the influence of higher-degree terms
}

#linear model
best_linear = grid_searching(LinearSVC(), param_grid_linear,X_train, X_test, y_train, y_test)

#RBF Model
best_rbf = grid_searching(SVC(kernel='rbf'), param_grid_rbf, X_train, X_test, y_train, y_test)

# Polynomial SVC
best_poly = grid_searching(SVC(kernel='poly'), param_grid_poly, X_train, X_test, y_train, y_test)

results = [    {**best_linear, 'Model': 'Linear SVC'},
    {**best_rbf, 'Model': 'RBF SVC'},
    {**best_poly, 'Model': 'Polynomial SVC'}]


# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\n=== Summary of All Models ===")
print(results_df[['Model', 'Test_Accuracy', 'Precision', 'Recall', 'F1', 'Training_Time']])

# === Simple and readable bar chart ===
metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1']

ax = results_df.set_index('Model')[metrics].plot(
    kind='bar',
    figsize=(12, 7),
    width=0.8
)

plt.title('Model Performance Comparison', fontsize=22, weight='bold')
plt.ylabel('Score', fontsize=18)
plt.xlabel('Model', fontsize=18)
plt.xticks(rotation=0, fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 1.0)  # this ensures the axis always ends at 1.0

plt.legend(title='Metrics', fontsize=14, title_fontsize=16, loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Simple bar chart for training time ===
results_df.set_index('Model')['Training_Time'].plot(kind='bar', figsize=(10, 6), color='orange')

plt.title('Training Time Comparison', fontsize=20)
plt.ylabel('Seconds', fontsize=16)
plt.xlabel('Model', fontsize=16)
plt.xticks(rotation=0, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
