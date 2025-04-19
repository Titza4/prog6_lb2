import pandas as pd
import numpy as np
import re
from collections import Counter

# Загрузка данных
df = pd.read_csv('train.csv')

print("=== Анализ данных Titanic ===\n")

# 1. Количество мужчин и женщин
sex_counts = df['Sex'].value_counts()
print(f"1. Количество мужчин и женщин на борту: {sex_counts.get('male', 0)}; {sex_counts.get('female', 0)}\n")

# 2. Количество пассажиров по портам
embarked_counts = df['Embarked'].value_counts()
print(f"2. Количество пассажиров, севших в портах: {embarked_counts.get('S', 0)}; {embarked_counts.get('C', 0)}; {embarked_counts.get('Q', 0)}\n")

# 3. Погибшие
total = len(df)
deaths = df['Survived'].value_counts().get(0, 0)
death_rate = deaths / total * 100
print(f"3. Погибших на параходе: {deaths} человек ({death_rate:.2f}%)\n")

# 4. Доли по классам
pclass_dist = df['Pclass'].value_counts(normalize=True) * 100
print("4. Доли пассажиров по классам:")
print(f"   1 класс: {pclass_dist.get(1, 0):.2f}%")
print(f"   2 класс: {pclass_dist.get(2, 0):.2f}%")
print(f"   3 класс: {pclass_dist.get(3, 0):.2f}%\n")

# 5. Корреляция SibSp и Parch
sibsp_parch_corr = df['SibSp'].corr(df['Parch'])
print(f"5. Корреляция между количеством супругов/родственников (SibSp) и детей (Parch): {sibsp_parch_corr:.3f}\n")

# 6. Корреляции с Survived
age_surv_corr = df[['Age', 'Survived']].dropna().corr().iloc[0, 1]
sex_numeric = df['Sex'].map({'male': 1, 'female': 0})
sex_surv_corr = sex_numeric.corr(df['Survived'])
pclass_surv_corr = df['Pclass'].corr(df['Survived'])

print("6. Корреляции с параметром выживания (Survived):")
print(f"   Возраст и выживание: {age_surv_corr:.3f}")
print(f"   Пол и выживание: {sex_surv_corr:.3f}")
print(f"   Класс и выживание: {pclass_surv_corr:.3f}\n")

# 7. Статистика по возрасту
age_stats = df['Age'].describe()
print("7. Возраст пассажиров:")
print(f"   Средний: {age_stats['mean']:.2f}")
print(f"   Медиана: {age_stats['50%']:.2f}")
print(f"   Минимальный: {age_stats['min']}")
print(f"   Максимальный: {age_stats['max']}\n")

# 8. Статистика по цене билета
fare_stats = df['Fare'].describe()
print("8. Цена билета:")
print(f"   Средняя: {fare_stats['mean']:.2f}")
print(f"   Медиана: {fare_stats['50%']:.2f}")
print(f"   Минимальная: {fare_stats['min']}")
print(f"   Максимальная: {fare_stats['max']}\n")

# 9. Самое популярное мужское имя на корабле
male_names = df[df['Sex'] == 'male']['Name']
male_first_names = []
for name in male_names:
    match = re.search(r'Mr\. ([A-Za-z]+)', name)
    if match:
        male_first_names.append(match.group(1))
male_name_counts = Counter(male_first_names)
most_common_male = male_name_counts.most_common(1)[0] if male_first_names else None

print("9. Самое популярное мужское имя на корабле:")
if most_common_male:
    print(f"   Имя: {most_common_male[0]}, Количество: {most_common_male[1]} человек\n")
else:
    print("   Данные отсутствуют\n")

# 10. Самые популярные имена среди людей старше 15 лет
df_over15 = df[df['Age'] > 15]

# Мужчины
male_names_over15 = df_over15[df_over15['Sex'] == 'male']['Name']
male_first_names_over15 = []
for name in male_names_over15:
    match = re.search(r'Mr\. ([A-Za-z]+)', name)
    if match:
        male_first_names_over15.append(match.group(1))
male_name_counts_over15 = Counter(male_first_names_over15)
most_common_male_over15 = male_name_counts_over15.most_common(1)[0] if male_first_names_over15 else None

# Женщины
female_names_over15 = df_over15[df_over15['Sex'] == 'female']['Name']
female_first_names_over15 = []
for name in female_names_over15:
    match = re.search(r'Miss\. ([A-Za-z]+)', name)
    if not match:
        match = re.search(r'Mrs\. [A-Za-z]+\s*\(([^)]+)\)', name)
        if match:
            candidate = match.group(1).split()[0]
            match = re.search(r'([A-Za-z]+)', candidate)
    if match:
        female_first_names_over15.append(match.group(1))
female_name_counts_over15 = Counter(female_first_names_over15)
most_common_female_over15 = female_name_counts_over15.most_common(1)[0] if female_first_names_over15 else None

print("10. Самые популярные имена среди людей старше 15 лет:")
if most_common_male_over15:
    print(f"   Мужское: {most_common_male_over15[0]}, Количество: {most_common_male_over15[1]} человек")
else:
    print("   Мужское: данные отсутствуют")
    
if most_common_female_over15:
    print(f"   Женское: {most_common_female_over15[0]}, Количество: {most_common_female_over15[1]} человек")
else:
    print("   Женское: данные отсутствуют")
