# Практика 16. Кореляційний аналіз — варіант 9 (Суми)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# 1. Генерація кліматичного набору даних для варіанта 9

np.random.seed(9)
city = "Суми"
monthly_temp = [-6, -5, 0, 9, 15, 18, 20, 19, 13, 7, 1, -4]
months = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер",
          "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]

rows = []
for month, temp in zip(months, monthly_temp):
    heating_days = np.clip(18 - temp + np.random.normal(0, 1.5), 0, 30)
    heating_days = round(heating_days)
    cost = 0.15 * heating_days + np.random.normal(0, 1.0)
    cost = round(max(cost, 0), 1)
    rows.append({
        "місто": city,
        "місяць": month,
        "температура": temp,
        "опалювальні_дні": heating_days,
        "витрати_на_опалення": cost,
    })

climate = pd.DataFrame(rows)
print("=" * 72)
print("ПРАКТИКА 16 — КОРЕЛЯЦІЙНИЙ АНАЛІЗ, ВАРІАНТ 9")
print("Місто:", city)
print("\nЗгенерований набір climate:")
print(climate.to_string(index=False))

# 2. Діаграма розсіювання та коефіцієнти Пірсона і Спірмена
x = climate["температура"]
y = climate["витрати_на_опалення"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=65)
for _, row in climate.iterrows():
    ax.annotate(row["місяць"], (row["температура"], row["витрати_на_опалення"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set_xlabel("Середньомісячна температура, °C")
ax.set_ylabel("Витрати на опалення, умовних тис. грн")
ax.set_title(f"{city}: температура та витрати на опалення")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("practice16.png", dpi=150)
plt.show()

pearson_result = pearsonr(x, y)
spearman_result = spearmanr(x, y)
ci = pearson_result.confidence_interval(confidence_level=0.95)

print("\n" + "=" * 72)
print("ЗАВДАННЯ 2. Коефіцієнти кореляції")
print(f"Пірсон:  r = {pearson_result.statistic:.4f}, p-value = {pearson_result.pvalue:.6g}")
print(f"Спірмен: rho = {spearman_result.statistic:.4f}, p-value = {spearman_result.pvalue:.6g}")
print(f"95%-й довірчий інтервал для r: [{ci.low:.4f}; {ci.high:.4f}]")

alpha = 0.05
print("\nІнтерпретація:")
if pearson_result.pvalue < alpha:
    print(f"- Для Пірсона p < {alpha}: нульову гіпотезу про нульову генеральну кореляцію відхиляємо.")
else:
    print(f"- Для Пірсона p ≥ {alpha}: немає достатніх підстав відхилити нульову гіпотезу.")
if spearman_result.pvalue < alpha:
    print(f"- Для Спірмена p < {alpha}: нульову гіпотезу про відсутність монотонного зв'язку відхиляємо.")
else:
    print(f"- Для Спірмена p ≥ {alpha}: немає достатніх підстав відхилити нульову гіпотезу.")
print("- Очікується негативний зв'язок: що нижча температура, то більшими зазвичай є витрати на опалення.")
print("- Довірчий інтервал оцінює правдоподібний діапазон значень істинної кореляції в генеральній сукупності за прийнятої моделі.")
print("- Якщо r і rho близькі, це узгоджується з приблизно монотонним зв'язком; відмінність може виникати через шум, округлення та невелику кількість спостережень.")

# 3. Кореляційні матриці Пірсона і Спірмена
numeric = climate[["температура", "опалювальні_дні", "витрати_на_опалення"]]
corr_pearson = numeric.corr()
corr_spearman = numeric.corr(method="spearman")

print("\n" + "=" * 72)
print("ЗАВДАННЯ 3. Кореляційна матриця Пірсона:")
print(corr_pearson.round(3).to_string())
print("\nКореляційна матриця Спірмена:")
print(corr_spearman.round(3).to_string())

# Визначення найсильнішої та найслабшої пари за абсолютним значенням
def correlation_pair_extremes(matrix):
    cols = list(matrix.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], matrix.iloc[i, j]))
    strongest = max(pairs, key=lambda p: abs(p[2]))
    weakest = min(pairs, key=lambda p: abs(p[2]))
    return strongest, weakest

strong_p, weak_p = correlation_pair_extremes(corr_pearson)
strong_s, weak_s = correlation_pair_extremes(corr_spearman)
print(f"\nПірсон — найсильніша пара: {strong_p[0]} / {strong_p[1]} (r={strong_p[2]:.3f}).")
print(f"Пірсон — найслабша пара: {weak_p[0]} / {weak_p[1]} (r={weak_p[2]:.3f}).")
print(f"Спірмен — найсильніша пара: {strong_s[0]} / {strong_s[1]} (rho={strong_s[2]:.3f}).")
print(f"Спірмен — найслабша пара: {weak_s[0]} / {weak_s[1]} (rho={weak_s[2]:.3f}).")
print("Це загалом очікувано: опалювальні дні обчислюються з температури з додаванням випадкового шуму,")
print("а витрати залежать від опалювальних днів і додаткового шуму. Тому змінні пов'язані формулою,")
print("але випадкові компоненти та округлення не дають кореляціям бути ідеальними.")

# 4. Кореляція і причинність
print("\n" + "=" * 72)
print("ЗАВДАННЯ 4. Кореляція і причинність")
print("У цьому навчальному наборі зв'язок не є лише випадковим збігом: код безпосередньо")
print("використовує температуру у формулі heating_days = 18 - temp + шум, а витрати")
print("обчислюються як 0.15 * heating_days + шум. Отже, у змодельованих даних закладено")
print("прямий механізм, через який нижча температура зазвичай збільшує витрати. На відміну")
print("від прикладу морозива й утоплень, де спекотна погода є спільним конфаундером, тут")
print("генератор не задає такого окремого конфаундера; однак це штучна модель, а не доказ")
print("реальної причинності в реальному місті, де можуть впливати тарифи, площа житла й утеплення.")

# 5. Додавання аномального спостереження та повторний аналіз
anomaly = pd.DataFrame([{
    "місто": city,
    "місяць": "Січ*",
    "температура": -6,
    "опалювальні_дні": 24,
    "витрати_на_опалення": round(float(climate.loc[0, "витрати_на_опалення"]) * 5, 1),
}])
climate_anomaly = pd.concat([climate, anomaly], ignore_index=True)

pearson_anomaly = pearsonr(climate_anomaly["температура"],
                           climate_anomaly["витрати_на_опалення"])
spearman_anomaly = spearmanr(climate_anomaly["температура"],
                             climate_anomaly["витрати_на_опалення"])

print("\n" + "=" * 72)
print("ЗАВДАННЯ 5. Аналіз після додавання аномального рядка")
print("Аномальний рядок:")
print(anomaly.to_string(index=False))
print(f"\nДо аномалії:    r = {pearson_result.statistic:.4f}; rho = {spearman_result.statistic:.4f}")
print(f"Після аномалії: r = {pearson_anomaly.statistic:.4f}; rho = {spearman_anomaly.statistic:.4f}")
print(f"Зміна Пірсона:  {abs(pearson_anomaly.statistic - pearson_result.statistic):.4f}")
print(f"Зміна Спірмена: {abs(spearman_anomaly.statistic - spearman_result.statistic):.4f}")
if abs(pearson_anomaly.statistic - pearson_result.statistic) > abs(spearman_anomaly.statistic - spearman_result.statistic):
    print("У цьому наборі сильніше змінився коефіцієнт Пірсона.")
elif abs(pearson_anomaly.statistic - pearson_result.statistic) < abs(spearman_anomaly.statistic - spearman_result.statistic):
    print("У цьому наборі сильніше змінився коефіцієнт Спірмена.")
else:
    print("Обидва коефіцієнти змінилися на однакову величину.")
print("Пірсон використовує фактичні значення та відхилення від середніх, тому велике аномальне")
print("значення може сильно змінити коваріацію і стандартне відхилення. Спірмен працює з рангами,")
print("тож величина викиду сама по собі не збільшує його ранг понад крайній; однак додавання")
print("спостереження все одно може змінити ранги та коефіцієнт.")

# Додаткова діаграма після аномалії
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(climate_anomaly["температура"], climate_anomaly["витрати_на_опалення"], s=65)
ax.annotate("Січ* — аномалія",
            (climate_anomaly.iloc[-1]["температура"],
             climate_anomaly.iloc[-1]["витрати_на_опалення"]),
            xytext=(8, 5), textcoords="offset points")
ax.set_xlabel("Середньомісячна температура, °C")
ax.set_ylabel("Витрати на опалення, умовних тис. грн")
ax.set_title(f"{city}: дані з аномальним спостереженням")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("practice16.png", dpi=150)
plt.show()
