from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import os

app = Flask(__name__)

class DataInsightProcessor:
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self.cleaned_data = None
        self.model = None
        self.feature_importances = None

    def preprocess_dataset(self):
        self.cleaned_data = self.raw_data.dropna()
        if 'income' in self.cleaned_data.columns and 'age' in self.cleaned_data.columns:
            self.cleaned_data['income_per_year'] = self.cleaned_data['income'] / (self.cleaned_data['age'] + 1)

    def perform_eda(self):
        print(self.cleaned_data.describe())
        sns.pairplot(self.cleaned_data)
        plt.show()

    def analyze_patterns(self):
        if {'income', 'age'}.issubset(self.cleaned_data.columns):
            clustering_model = KMeans(n_clusters=4)
            self.cleaned_data['cluster_group'] = clustering_model.fit_predict(self.cleaned_data[['income', 'age']])
        if 'target' in self.cleaned_data.columns:
            features = self.cleaned_data.drop('target', axis=1)
            target = self.cleaned_data['target']
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            self.feature_importances = pd.Series(self.model.feature_importances_, index=features.columns)

    def extract_insights(self):
        if self.feature_importances is not None:
            print(self.feature_importances.sort_values(ascending=False))

    def visualize_insights(self):
        if self.feature_importances is not None:
            sns.barplot(x=self.feature_importances.index, y=self.feature_importances.values)
            plt.title('Important Features')
            plt.show()

    def run_all_steps(self):
        self.preprocess_dataset()
        self.perform_eda()
        self.analyze_patterns()
        self.extract_insights()
        self.visualize_insights()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run():
    processor = DataInsightProcessor('C:/Users/ISHAN/OneDrive/Desktop/project/intel/income.csv')
    processor.run_all_steps()
    return 'Processing complete! Check the console for output.'

if __name__ == '__main__':
    app.run(debug=True)
