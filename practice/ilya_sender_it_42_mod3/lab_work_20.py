import numpy as np
from scipy.optimize import linprog


# Варіант 9 — Суми

city = "Суми"

# Дані варіанта
profit_A = 46
profit_B = 60

raw_A = 3
raw_B = 2
raw_limit = 145

time_A = 2
time_B = 5
time_limit = 225

electric_A = 2
electric_B = 1
electric_limit = 105


# Завдання 1. Математична постановка задачі

print("Завдання 1. Математична постановка задачі")
print("Місто:", city)
print("x1 — кількість виробів А за тиждень")
print("x2 — кількість виробів Б за тиждень")
print()
print("Максимізувати прибуток:")
print("P = 46*x1 + 60*x2")
print()
print("Обмеження:")
print("3*x1 + 2*x2 <= 145  (сировина, кг)")
print("2*x1 + 5*x2 <= 225  (робочий час, год)")
print("2*x1 + x2 <= 105    (електроенергія, кВт·год)")
print("x1 >= 0, x2 >= 0")


# Коефіцієнти цільової функції.
# linprog виконує мінімізацію, тому прибуток записуємо
# з від'ємними коефіцієнтами.

c = [-profit_A, -profit_B]

# Матриця коефіцієнтів ресурсних обмежень
A_ub = [
    [raw_A, raw_B],
    [time_A, time_B],
    [electric_A, electric_B]
]

# Максимальні запаси ресурсів
b_ub = [
    raw_limit,
    time_limit,
    electric_limit
]

bounds = [(0, None), (0, None)]


# Завдання 2. Розв'язання через linprog

res = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method="highs"
)

print("\nЗавдання 2. Результат оптимізації")

print("res.x =", res.x)
print("res.fun =", res.fun)
print("Максимальний прибуток =", -res.fun, "грн")
print("res.status =", res.status)
print("res.message =", res.message)

if res.status == 0:
    print("Оптимальний розв'язок знайдено успішно.")
else:
    print("Оптимальний розв'язок не знайдено.")


# Завдання 3. Інтерпретація оптимального розв'язку

print("\nЗавдання 3. Інтерпретація результату")

if res.status == 0:
    x1, x2 = res.x
    max_profit = -res.fun

    print("Оптимальна кількість виробів А =", round(x1, 4))
    print("Оптимальна кількість виробів Б =", round(x2, 4))
    print("Максимальний тижневий прибуток =", round(max_profit, 2), "грн")

    print(
        "За оптимальним планом потрібно виробляти",
        round(x1, 4), "виробів А та",
        round(x2, 4), "виробів Б на тиждень."
    )

    if x1.is_integer() and x2.is_integer():
        print("Кількість виробів є цілою.")
    else:
        print(
            "Лінійна модель допускає дробові значення кількості виробів. "
            "Для реального виробництва ці кількості можуть бути недопустимими. "
            "Звичайне округлення не гарантує оптимального цілочислового "
            "розв'язку, тому для точного планування потрібна цілочислова "
            "оптимізація."
        )


# Завдання 4. Аналіз активних обмежень

print("\nЗавдання 4. Активні та неактивні обмеження")

resource_names = [
    "Сировина",
    "Робочий час",
    "Електроенергія"
]

resource_units = [
    "кг",
    "год",
    "кВт·год"
]

resource_limits = np.array(b_ub, dtype=float)

if res.status == 0:
    x1, x2 = res.x

    resource_usage = np.array([
        raw_A * x1 + raw_B * x2,
        time_A * x1 + time_B * x2,
        electric_A * x1 + electric_B * x2
    ])

    slack = res.slack

    print("Використання ресурсів:", resource_usage)
    print("Залишки ресурсів res.slack:", slack)

    for i, name in enumerate(resource_names):
        print("\nРесурс:", name)
        print("Використано:", round(resource_usage[i], 4), resource_units[i])
        print("Ліміт:", resource_limits[i], resource_units[i])
        print("Запас:", round(slack[i], 4), resource_units[i])

        if np.isclose(slack[i], 0, atol=1e-7):
            print("Обмеження активне: ресурс використано повністю.")
        else:
            print("Обмеження неактивне: ресурс має невикористаний запас.")

    print(
        "\nІнтерпретація: активні обмеження визначають межі виробництва. "
        "Збільшення запасу активного ресурсу може підвищити прибуток. "
        "Збільшення неактивного ресурсу саме по собі не обов'язково "
        "покращить результат, оскільки він не є поточним вузьким місцем."
    )


# Завдання 5. What-if: зміна ресурсних обмежень


# Частина 1. Збільшення ліміту сировини на 15%

print("\nЗавдання 5.1. Збільшення запасу сировини на 15%")

b_raw_more = b_ub.copy()
b_raw_more[0] = raw_limit * 1.15

res_raw_more = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_raw_more,
    bounds=bounds,
    method="highs"
)

if res_raw_more.status == 0:
    print("Новий ліміт сировини =", b_raw_more[0], "кг")
    print("Новий res.x =", res_raw_more.x)
    print("Новий res.fun =", res_raw_more.fun)
    print("Новий максимальний прибуток =", -res_raw_more.fun, "грн")

    print(
        "Зміна прибутку =",
        round(-res_raw_more.fun - (-res.fun), 2),
        "грн"
    )

    print("Початковий план =", res.x)
    print("Новий план =", res_raw_more.x)

    if not np.allclose(res_raw_more.x, res.x):
        print("Структура виробництва змінилася.")
    else:
        print("Структура виробництва не змінилася.")


# Частина 2. Збільшення ліміту електроенергії на 50%

print("\nЗавдання 5.2. Збільшення запасу електроенергії на 50%")

b_electric_more = b_ub.copy()
b_electric_more[2] = electric_limit * 1.50

res_electric_more = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_electric_more,
    bounds=bounds,
    method="highs"
)

if res_electric_more.status == 0:
    print("Новий ліміт електроенергії =", b_electric_more[2], "кВт·год")
    print("Новий res.x =", res_electric_more.x)
    print("Новий res.fun =", res_electric_more.fun)
    print("Новий максимальний прибуток =", -res_electric_more.fun, "грн")

    print(
        "Зміна прибутку =",
        round(-res_electric_more.fun - (-res.fun), 2),
        "грн"
    )

    if np.allclose(res_electric_more.x, res.x):
        print(
            "Прибуток не змінився, оскільки обмеження електроенергії "
            "було неактивним і не визначало оптимальний план."
        )
    else:
        print(
            "Оптимальний план змінився. Це означає, що після зміни "
            "ліміту змінилися умови оптимізації."
        )
