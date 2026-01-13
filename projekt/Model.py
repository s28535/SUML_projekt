import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pycaret.regression import *

class ModelMyNN(nn.Module):
    def __init__(self, in_features=27, h1=27, h2=27, h3=27, out_features=1):
        super().__init__()

        self.criterion = nn.MSELoss()
        self.losses = []


        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2 ,h3)
        self.out = nn.Linear(h3,
                             out_features)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

    def trainModel(self,x_train,y_train,x_test,y_test,epoches,scaler_X,scaler_y):
        #Wczytanie danych
        x_train = torch.FloatTensor(x_train.to_numpy())
        y_train = torch.FloatTensor(y_train.to_numpy())
        x_test = torch.FloatTensor(x_test.to_numpy())
        y_test = torch.FloatTensor(y_test.to_numpy())


        print("x_train: ", x_train[:5])
        print("y_train: ", y_train[:5])
        print("x_test: ", x_test[:5])
        print("y_test: ", y_test[:5])

        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)

        for i in range(epoches):
            y_pred = self.forward(x_train)
            loss = self.criterion(y_pred, y_train)
            self.losses.append(loss.detach().numpy())

            if i % 10 == 0:
                print("Epoka: ", i, ", błąd: ", loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        plt.plot(self.losses)
        plt.title('Loss over generations')
        plt.savefig('data/Charts/lossesOverTime.png')
        plt.show()

        Path("data/models").mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = f"data/models/MyMM_{timestamp}.pkl"

        torch.save(self.state_dict(), model_path)
        print(" Model zapisanu: ",model_path)

class AutoML():
    def testDifrentMLM(self, df):
        # Podział na zbiór treningowy i testowy
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        print("Treningowy zbiór danych:")
        print(train_df.head())

        # PyCaret setup
        setup(data=train_df, target='price', session_id=42)

        # Porównanie modeli
        best = compare_models()
        print(f"Najlepszy model: {best}")

        # Generowanie wykresów
        plots = [
                 'feature', 'feature_all']

        Path("data/Reporting").mkdir(parents=True, exist_ok=True)
        for p in plots:

            plot_model(best, plot=p, save=True, display_format='streamlit')



        # Finalizacja modelu
        final_model = finalize_model(best)

        # Walidacja krzyżowa
        cv_scores = cross_val_score(final_model, train_df.drop('price', axis=1),
                                    train_df['price'], cv=5, scoring='r2')
        print(f"Walidacja krzyżowa: {cv_scores}, średnia: {np.mean(cv_scores)}")

        # Testowanie na zbiorze testowym
        X_test = test_df.drop('price', axis=1)
        y_test = test_df['price']
        y_pred = final_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"Test Metrics -> RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

        # Zapis wyników do JSON
        test_metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": float(mape)
        }

        with open('data/reporting/test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)


        Path("data/models").mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = f"data/models/AutoML_{timestamp}"
        save_model(final_model, model_path)
        print("Model zapisany:", model_path)