import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Варіант 9 — Суми
np.random.seed(42)

city = "Суми"
mean_temp = 7.5
amplitude = 14
hum_base = 73

rows = []

for month in range(1, 13):
    temp = (
        mean_temp
        + amplitude * np.cos((month - 7) / 12 * 2 * np.pi)
        + np.random.normal(0, 1.0)
    )

    humidity = hum_base + np.random.normal(0, 5.0)

    if temp < 8:
        opal_days = 30
    elif temp < 12:
        opal_days = 15
    else:
        opal_days = 0

    cost = (
        200
        - 9.0 * temp
        + 0.6 * humidity
        + 2.5 * opal_days
        + np.random.normal(0, 15)
    )

    rows.append({
        "місто": city,
        "місяць": month,
        "температура": round(temp, 1),
        "вологість": round(humidity, 1),
        "опалювальні_дні": opal_days,
        "витрати_на_опалення": round(cost, 1),
    })

heating = pd.DataFrame(rows)

print("Вихідний набір даних:")
print(heating)


# Завдання 1. Модель із двома предикторами

X = sm.add_constant(
    heating[["температура", "вологість"]]
)
y = heating["витрати_на_опалення"]

model = sm.OLS(y, X).fit()

print("\nЗавдання 1")
print(model.summary())

print("R² =", model.rsquared)
print("Adjusted R² =", model.rsquared_adj)


# Завдання 2. Інтерпретація коефіцієнтів

print("\nЗавдання 2")

for col in ["температура", "вологість"]:
    coef = model.params[col]
    p_value = model.pvalues[col]

    print("\nПредиктор:", col)
    print("Коефіцієнт =", round(coef, 4))
    print("P-value =", round(p_value, 4))

    if p_value < 0.05:
        print("Предиктор статистично значущий при alpha = 0.05")
    else:
        print("Немає достатніх підстав вважати предиктор статистично значущим")


# Завдання 3. Перевірка мультиколінеарності через VIF

print("\nЗавдання 3")

for i, col in enumerate(X.columns):
    if col == "const":
        continue

    vif = variance_inflation_factor(X.values, i)
    print(col, "VIF =", round(vif, 2))


# Завдання 4. Модель із трьома предикторами

X3 = sm.add_constant(
    heating[["температура", "вологість", "опалювальні_дні"]]
)

model3 = sm.OLS(y, X3).fit()

print("\nЗавдання 4")
print(model3.summary())

print("\nПорівняння моделей:")
print("Модель із двома предикторами:")
print("R² =", round(model.rsquared, 4))
print("Adjusted R² =", round(model.rsquared_adj, 4))

print("\nМодель із трьома предикторами:")
print("R² =", round(model3.rsquared, 4))
print("Adjusted R² =", round(model3.rsquared_adj, 4))

print("\nVIF моделі з трьома предикторами:")

for i, col in enumerate(X3.columns):
    if col == "const":
        continue

    vif = variance_inflation_factor(X3.values, i)
    print(col, "VIF =", round(vif, 2))


# Завдання 5. Діагностика залишків

fitted = model.fittedvalues
residuals = model.resid

print("\nЗавдання 5")
print(
    "Кореляція залишків із передбаченими значеннями =",
    np.corrcoef(fitted, residuals)[0, 1]
)

# Графік залишків проти передбачених значень

plt.figure(figsize=(8, 5))
plt.scatter(fitted, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Передбачені витрати на опалення")
plt.ylabel("Залишки")
plt.title("Залишки проти передбачених значень")
plt.grid(True)
plt.show()

# QQ-plot залишків

plt.figure(figsize=(8, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-plot залишків")
plt.grid(True)
plt.show()

# Критерій Шапіро — Вілка

shapiro_stat, shapiro_p = stats.shapiro(residuals)

print("\nКритерій Шапіро — Вілка")
print("W =", round(shapiro_stat, 4))
print("p-value =", round(shapiro_p, 4))

if shapiro_p < 0.05:
    print("Відхиляємо H0: нормальність залишків не підтверджується")
else:
    print("Не відхиляємо H0: немає достатніх підстав заперечувати нормальність залишків")


# Завдання 6. Порівняння з простою регресією

X_simple = sm.add_constant(heating[["температура"]])
model_simple = sm.OLS(y, X_simple).fit()

print("\nЗавдання 6")

print("Проста регресія — тільки температура:")
print(model_simple.summary())

print("R² простої регресії =", round(model_simple.rsquared, 4))
print("R² множинної регресії =", round(model.rsquared, 4))
print("Adjusted R² множинної регресії =", round(model.rsquared_adj, 4))

print(
    "Зміна R² після додавання вологості =",
    round(model.rsquared - model_simple.rsquared, 4)
)


