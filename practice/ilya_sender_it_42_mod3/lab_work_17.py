# Практика 17. Проста лінійна регресія — варіант 9 (Суми)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

# 1. Генерація того самого climate, що й у Практиці 16
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
x = climate["температура"].to_numpy(dtype=float)
y = climate["витрати_на_опалення"].to_numpy(dtype=float)

print("=" * 76)
print("ПРАКТИКА 17 — ПРОСТА ЛІНІЙНА РЕГРЕСІЯ, ВАРІАНТ 9")
print("Місто:", city)
print("\nНабір climate:")
print(climate.to_string(index=False))

# Завдання 1. Регресійна модель
result = linregress(x, y)
b0 = result.intercept
b1 = result.slope
predicted = b0 + b1 * x
residuals = y - predicted

print("\n" + "=" * 76)
print("ЗАВДАННЯ 1. Побудова регресійної моделі")
print(f"Перетин b0 = {b0:.3f}")
print(f"Нахил b1 = {b1:.3f}")
print(f"Рівняння: витрати ≈ {b0:.3f} + ({b1:.3f}) × температура")
print(f"R² = {result.rvalue ** 2:.4f}")
print(f"p-value для нахилу = {result.pvalue:.6g}")
print(f"Стандартна похибка нахилу = {result.stderr:.4f}")
print("Інтерпретація: при підвищенні середньомісячної температури на 1 °C")
print(f"модель прогнозує зміну витрат приблизно на {b1:.3f} умовних тис. грн.")
print("Негативний нахил узгоджується з очікуванням: у тепліші місяці опалення")
print("зазвичай потребує менше витрат. Перетин b0 — формальний прогноз при 0 °C.")

# Завдання 2. R² двома способами та зв'язок із r²
ss_res = np.sum((y - predicted) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_formula = 1 - ss_res / ss_tot
r2_rvalue = result.rvalue ** 2
pearson_result = pearsonr(x, y)
r2_pearson = pearson_result.statistic ** 2

print("\n" + "=" * 76)
print("ЗАВДАННЯ 2. Коефіцієнт детермінації")
print(f"SS залишків = {ss_res:.4f}")
print(f"SS загальна = {ss_tot:.4f}")
print(f"R² через result.rvalue ** 2 = {r2_rvalue:.6f}")
print(f"R² через 1 - SS_res / SS_tot = {r2_formula:.6f}")
print(f"r Пірсона = {pearson_result.statistic:.6f}; r² = {r2_pearson:.6f}")
print("У простій лінійній регресії з одним предиктором R² = r² Пірсона.")
print(f"Температура пояснює приблизно {100*r2_formula:.1f}% вибіркового розкиду витрат;")
print(f"приблизно {100*(1-r2_formula):.1f}% розкиду лишається непоясненим моделлю.")

# Завдання 3. Таблиця залишків і графіки
climate["передбачені_витрати"] = predicted
climate["залишок"] = residuals
print("\n" + "=" * 76)
print("ЗАВДАННЯ 3. Передбачені значення та залишки")
print(climate[["місяць", "температура", "витрати_на_опалення",
               "передбачені_витрати", "залишок"]].round(3).to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(x, y, s=65, label="Фактичні дані")
x_line = np.linspace(x.min(), x.max(), 200)
ax1.plot(x_line, b0 + b1 * x_line, label="Регресійна пряма")
ax1.set_xlabel("Середньомісячна температура, °C")
ax1.set_ylabel("Витрати на опалення, умовних тис. грн")
ax1.set_title("Регресія: температура та витрати")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.scatter(x, residuals, s=65)
for month, temp, resid in zip(months, x, residuals):
    ax2.annotate(month, (temp, resid), xytext=(4, 4),
                 textcoords="offset points", fontsize=8)
ax2.axhline(0, linestyle="--", linewidth=1)
ax2.set_xlabel("Середньомісячна температура, °C")
ax2.set_ylabel("Залишок (факт − прогноз)")
ax2.set_title("Графік залишків")
ax2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("practice17.png", dpi=150)
plt.show()

warm = climate[climate["температура"] >= 15][["місяць", "температура",
                                                "витрати_на_опалення",
                                                "передбачені_витрати", "залишок"]]
print("\nНай тепліші місяці (температура ≥ 15 °C):")
print(warm.round(3).to_string(index=False))
print("Інтерпретація графіка залишків: оцініть, чи залишки розташовані випадково навколо нуля.")
print("Для цього набору нульове обмеження витрат може давати додатні залишки в теплі місяці,")
print("якщо лінійна модель прогнозує від'ємні або дуже малі витрати, а фактичні витрати не нижчі нуля.")
print("Така систематична структура означає, що лінійність не повністю описує дані; саме тому")
print("високого R² недостатньо без перевірки графіка залишків.")

# Завдання 4. Прогноз при -10 °C та екстраполяція
x_new = -10.0
y_new = b0 + b1 * x_new
x_min, x_max = float(np.min(x)), float(np.max(x))
inside = x_min <= x_new <= x_max

print("\n" + "=" * 76)
print("ЗАВДАННЯ 4. Прогноз та екстраполяція")
print(f"Діапазон температур у даних: від {x_min:.1f} °C до {x_max:.1f} °C")
print(f"Прогноз витрат при {x_new:.1f} °C: {y_new:.3f} умовних тис. грн")
if inside:
    print("Це інтерполяція: температура лежить усередині спостереженого діапазону.")
else:
    print("Це екстраполяція: температура лежить за межами спостереженого діапазону.")
    print("Надійність такого прогнозу нижча, бо лінійна залежність не перевірена за цією межею;")
    print("поза діапазоном можуть діяти інші умови, а пряма може давати фізично недоречні значення.")

# Завдання 5. Порівняння p-value регресії та кореляції
print("\n" + "=" * 76)
print("ЗАВДАННЯ 5. Порівняння з коефіцієнтом кореляції")
print(f"p-value linregress (H0: нахил = 0) = {result.pvalue:.8g}")
print(f"p-value pearsonr (H0: кореляція = 0) = {pearson_result.pvalue:.8g}")
print("У простій лінійній регресії з одним предиктором ці перевірки еквівалентні:")
print("ненульовий нахил відповідає ненульовому лінійному зв'язку, тому p-value однакові")
print("або відрізняються лише через числову точність реалізації.")

