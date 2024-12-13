import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------------- Читання файлів Excel -------------------------------------------
data_description = pd.read_excel("data_description.xlsx")
sample_data = pd.read_excel("sample_data.xlsx")

print("Опис даних:")
print(data_description.head())

print("Приклад даних:")
print(sample_data.head())


# --------------------------------- Вибір індикаторів та підготовка даних ---------------------------------
indicators = [
    "loan_amount",
    "loan_days",
    "product_profile_id",
    "credit_policy_id",
    "user_id",
    "prolongation_number",
    "prolongation_total_days",
    "wizard_type_id",
    "step",
]
X = sample_data[indicators]

# Створення бінарної мети: 1 - кредит повернуто, 0 - кредит не повернуто
sample_data["target"] = sample_data["closed_at"].notnull().astype(int)
y = sample_data["target"]

# Розділення даних на навчальні та тестові
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Навчання моделі дерева прийняття рішень
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Прогнозування
y_pred = clf.predict(X_test)

# Оцінка моделі з використанням параметра zero_division
print("Точність моделі:", accuracy_score(y_test, y_pred))
print("Звіт класифікації:")
print(classification_report(y_test, y_pred, zero_division=1))

# Візуалізація результатів за допомогою матриці змішування
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Матриця змішування для моделі дерева прийняття рішень")
plt.show()

# --------------------------------- Виявлення аномальних значень ---------------------------------
threshold = 3
mean_extension = sample_data["loan_amount"].mean()
std_extension = sample_data["loan_amount"].std()
outliers = sample_data[
    np.abs(sample_data["loan_amount"] - mean_extension) > threshold * std_extension
]

print("Аномальні значення:")
print(outliers)

# Візуалізація аномальних значень
plt.figure(figsize=(10, 6))
plt.scatter(sample_data.index, sample_data["loan_amount"], label="Нормальні значення")
plt.scatter(outliers.index, outliers["loan_amount"], color="r", label="Аномалії")
plt.xlabel("Індекс")
plt.ylabel("Сума кредита")
plt.title("Виявлення аномальних значень")
plt.legend()
plt.show()

# Запис аномальних значень у файл Excel
outliers.to_excel("anomalies.xlsx", index=False)
print("Аномальні значення збережені у файл anomalies.xlsx")
