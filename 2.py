import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

data = pd.read_csv("v_3.csv", sep=';', encoding="cp1251")
data = pd.DataFrame(data)
print("Первые 5 строк данных:")
print(data.head())
print("\nИнформация о структуре данных:")
print(data.info())

new_column_names = ['№п/п', 'Группа', 'Пол', 'Возраст','Стаж','Доход','Специализация', 'Страницы', 'Активность'] 
data.columns = new_column_names 
print(data.info())

categorical_columns = [
    "Группа",
    "Пол",
    "Специализация"
]

for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].astype("category")
        print(f"Столбец '{col}' преобразован в категориальный.")

print("\nТипы данных после преобразования:")
print(data.dtypes)
print(data.info())

# -----------------------------------------------------------------------------
#                                  Пункт 5
# -----------------------------------------------------------------------------


def signif_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

def ggpairs_like(df, features, hue='Группа'):
    n = len(features)
    classes = sorted(df[hue].unique())
    
    colors = ['#1f77b4a0', '#ff7f0ea0', '#2ca02ca0', '#d62728a0']
    color_map = dict(zip(classes, colors[:len(classes)]))
    
    fig, axes = plt.subplots(n, n, figsize=(2*n, 2*n))
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            xvar, yvar = features[j], features[i]

            if i < n-1:
                ax.tick_params(bottom=False, labelbottom=False)
            else:
                ax.tick_params(bottom=True, labelbottom=True)

            if j > 0:
                ax.tick_params(left=False, labelleft=False)
            else:
                ax.tick_params(left=True, labelleft=True)
                ax.tick_params(axis='y', labelsize=8)

            if i == j:
                if pd.api.types.is_numeric_dtype(df[xvar]):
                    for cls in classes:
                        cls_data = df[df[hue] == cls][xvar]
                        sns.kdeplot(data=cls_data, fill=True, alpha=0.6, 
                                   color=color_map[cls], ax=ax, label=cls)
                    ax.legend().remove()
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    ax.set_ylabel('')
                else:
                    sns.countplot(data=df, x=xvar, hue=hue, ax=ax, palette=color_map)
                    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                    ax.set_ylabel('')
                    
            elif i > j:
                if pd.api.types.is_numeric_dtype(df[xvar]) and pd.api.types.is_numeric_dtype(df[yvar]):
                    for cls in classes:
                        cls_data = df[df[hue] == cls]
                        ax.scatter(cls_data[xvar], cls_data[yvar], alpha=0.6, s=12,
                                 color=color_map[cls], label=cls)
                    ax.legend().remove()
                    ax.grid(True, linestyle='--', alpha=0.3)
                else:
                    ax.axis('off')
            else:
                if pd.api.types.is_numeric_dtype(df[xvar]) and pd.api.types.is_numeric_dtype(df[yvar]):
                    valid_data = df[[xvar, yvar]].dropna()
                    if len(valid_data) > 2:
                        r, p = pearsonr(valid_data[xvar], valid_data[yvar])
                        
                        y_pos = 0.85
                        ax.text(0.5, y_pos, f"Corr: {r:.3f}{signif_stars(p)}", 
                               ha='center', va='center', fontsize=14, transform=ax.transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
                        
                        y_pos -= 0.20
                        for cls_idx, cls in enumerate(classes):
                            sub = df[df[hue]==cls]
                            valid_sub = sub[[xvar, yvar]].dropna()
                            if len(valid_sub) > 2:
                                r_c, p_c = pearsonr(valid_sub[xvar], valid_sub[yvar])
                                ax.text(0.5, y_pos, f"{cls}: {r_c:.3f}{signif_stars(p_c)}", 
                                       ha='center', va='center', fontsize=12, 
                                       color=color_map[cls], transform=ax.transAxes,
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
                            y_pos -= 0.18
                    else:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center', 
                               fontsize=12, transform=ax.transAxes)
                ax.axis('off')

            if i == 0 and j != 0:
                if pd.api.types.is_numeric_dtype(df[xvar]):
                    boxplot_data = [df[df[hue] == cls][xvar].dropna() for cls in classes]
                    bp = ax.boxplot(boxplot_data, patch_artist=True)
                    
                    for patch, cls in zip(bp['boxes'], classes):
                        patch.set_facecolor(color_map[cls])
                        patch.set_alpha(0.7)
                    
                    ax.set_xticks(range(1, len(classes) + 1))
                    ax.set_xticklabels(classes, fontsize=8)
                    
                    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                    ax.set_ylabel('')

            if j == 0 and i != 0:
                if pd.api.types.is_numeric_dtype(df[yvar]):
                    ax.clear()
                    
                    n_classes = len(classes)
                    
                    y_min, y_max = df[yvar].min(), df[yvar].max()
                    bins = np.histogram_bin_edges(df[yvar], bins='auto')
                    
                    max_count = 0
                    class_histograms = []
                    
                    for cls in classes:
                        cls_data = df[df[hue] == cls][yvar]
                        hist, _ = np.histogram(cls_data, bins=bins)
                        max_count = max(max_count, hist.max())
                        class_histograms.append(hist)
                    
                    max_count *= 1.1
                    
                    total_width = 0.9
                    class_width = total_width / n_classes
                    
                    for cls_idx, cls in enumerate(classes):
                        hist = class_histograms[cls_idx]
                        x_left = (1.0 - total_width) / 2 + cls_idx * class_width
                        
                        for bin_idx in range(len(hist)):
                            bin_value = hist[bin_idx]
                            if bin_value > 0:
                                y_bottom = bins[bin_idx]
                                y_top = bins[bin_idx + 1]
                                
                                height = (bin_value / max_count) * class_width * 0.95
                                
                                rect = plt.Rectangle((x_left, y_bottom), height, y_top - y_bottom,
                                                   facecolor=color_map[cls], alpha=0.8, 
                                                   edgecolor='white', linewidth=0.5)
                                ax.add_patch(rect)
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(y_min, y_max)
                    ax.set_ylabel('')
                    
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    
                    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                    
                    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                    ax.tick_params(axis='y', labelsize=8)  

            ax.grid(True, linestyle='--', alpha=0.3)

    for j in range(n):
        axes[0, j].set_title(features[j].upper(), fontsize=11, fontweight='bold', pad=15)
    
    for i in range(n):
        axes[i, n-1].text(1.15, 0.5, features[i].upper(), fontsize=11, fontweight='bold',
                        ha='left', va='center', rotation=270, transform=axes[i, n-1].transAxes)

    axes[n-1, n-1].set_xlabel('')

    plt.tight_layout()
    plt.show()

ggpairs_like(data, ['Группа', 'Возраст','Стаж','Доход', 'Страницы', 'Активность'])
ggpairs_like(data, ['Пол', 'Возраст','Стаж','Доход','Страницы', 'Активность'])
ggpairs_like(data, ['Специализация', 'Возраст','Стаж','Доход','Страницы', 'Активность'])


# -----------------------------------------------------------------------------
#                                  Пункт 5.1
# -----------------------------------------------------------------------------


import statsmodels.api as sm 

df_group1 = data[data['Группа'] == 1].copy()

X = df_group1[['Возраст', 'Стаж', 'Доход', 'Страницы', ]]
y = df_group1['Активность']

X_with_const = sm.add_constant(X)

print("Матрица признаков X с константой:")
print(X_with_const)
print("\nЦелевая переменная y:")
print(y)

model = sm.OLS(y, X_with_const)

results = model.fit()

print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 1 =============================")
print(results.summary())

intercept = results.params['const']
coef_v = results.params['Возраст']
coef_s = results.params['Стаж']
coef_r = results.params['Доход']
coef_o = results.params['Страницы']

print(f"Уравнение регрессии:")
print(f"Активность = {intercept:.2f} + {coef_v:.2f} * Возраст + {coef_s:.2f} * Стаж + {coef_r:.2f} * Доход + {coef_o:.2f} * Страницы")

residuals = results.resid

df_group1['Предсказания'] = results.predict(X_with_const)
df_group1['Остатки'] = residuals

print(df_group1)

# Графики для 5.1
# График остатков для проверки гомоскедастичности
plt.figure(figsize=(8, 6))
plt.scatter(df_group1['Предсказания'], df_group1['Остатки'])
plt.axhline(y=0, color='r', linestyle='--') 
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков для проверки гомоскедастичности')
plt.show()

# Подграфики
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Гистограмма с кривой нормального распределения
sns.histplot(residuals, kde=True, ax=ax[0])
ax[0].set_title('Гистограмма остатков')
ax[0].set_xlabel('Остатки')

# График Q-Q
stats.probplot(residuals, dist="norm", plot=ax[1])
ax[1].set_title('График Q-Q для проверки нормальности')
ax[1].set_xlabel('Теоретические квантили')
ax[1].set_ylabel('Квантили выборки')

plt.tight_layout()
plt.show()

# График: Реальные значения и Предсказанные
predictions = results.predict(X_with_const)

plt.figure(figsize=(8, 6))

plt.scatter(predictions, y, color='blue', alpha=0.7, edgecolors='w', s=80)

min_val = min(min(predictions), min(y))
max_val = max(max(predictions), max(y))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальная линия')

plt.xlabel('Предсказанные значения активности', fontsize=12)
plt.ylabel('Реальные значения активности', fontsize=12)
plt.title('График: Реальные значения и Предсказанные', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Базовые статистики распределения остатков
min_resid = residuals.min()
max_resid = residuals.max()
sd_resid = residuals.std(ddof=1) 
me = residuals.mean() 
mpe = (residuals / y).mean() * 100 

mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))
r2 = results.rsquared
mape = np.mean(np.abs((y - predictions) / y)) * 100

print("Рассчитанные характеристики:")
print(f"Min (минимальный остаток): {min_resid:,.0f}".replace(',', ' '))
print(f"Max (максимальный остаток): {max_resid:,.0f}".replace(',', ' '))
print(f"SD (стандартное отклонение остатков): {sd_resid:,.0f}".replace(',', ' '))
print(f"ME (средняя ошибка): {me:.2f}")
print(f"MPE (средняя процентная ошибка): {mpe:.2f}%")
print(f"MAE (средняя абсолютная ошибка): {mae:,.0f}".replace(',', ' '))
print(f"RMSE (корень из СКО): {rmse:,.0f}".replace(',', ' '))
print(f"MAPE (ср. абс. % ошибка): {mape:.2f}%")
print(f"R2 (коэффициент детерминации): {r2:.4f}")

# -----------------------------------------------------------------------------
#                                  Пункт 5.2
# -----------------------------------------------------------------------------

X_with_const = sm.add_constant(X)

SL = 0.05

model = sm.OLS(y, X_with_const)
results = model.fit()

print("Начальная модель со всеми признаками:")
print(results.summary())

num_vars = len(X.columns)
for i in range(num_vars):
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    max_p_value = max(results.pvalues[1:]) 
    
    if max_p_value > SL:
        j = np.argmax(results.pvalues[1:]) + 1 
        X_with_const = X_with_const.drop(X_with_const.columns[j], axis=1)
    else:
        break

print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 2 =============================")
final_results = sm.OLS(y, X_with_const).fit()
print(final_results.summary())

predictions2 = final_results.predict(X_with_const)
residuals2 = y - predictions2

min_resid_m2 = residuals2.min()
max_resid_m2 = residuals2.max()
sd_resid_m2 = residuals2.std(ddof=1) 
me_m2 = residuals2.mean() 
mpe_m2 = (residuals2 / y).mean() * 100 

mae_m2 = np.mean(np.abs(residuals2))
rmse_m2 = np.sqrt(np.mean(residuals2**2))
r2_m2 = final_results.rsquared
mape_m2 = np.mean(np.abs((y - predictions2) / y)) * 100

print("\nХарактеристики точности для Модели 2 (в Таблицу 1):")
print(f"ME (средняя ошибка): {me_m2:.2f}")
print(f"Min (минимальный остаток): {min_resid_m2:.2f}")
print(f"Max (максимальный остаток): {max_resid_m2:.2f}")
print(f"MAE (средняя абсолютная ошибка): {mae_m2:.2f}")
print(f"SD (стандартное отклонение остатков): {sd_resid_m2:.2f}")
print(f"MPE (средняя процентная ошибка): {mpe_m2:.2f}%")
print(f"MAPE (ср. абс. % ошибка): {mape_m2:.2f}%")
print(f"MSE (среднеквадратическая ошибка): {np.mean(residuals2**2):.2f}")
print(f"RMSE (корень из СКО): {rmse_m2:.2f}")
print(f"D (коэффициент детерминации): {r2_m2:.4f}")
print(f"AIC (критерий Акаике): {final_results.aic:.2f}")
print(f"BIC (критерий Шварца): {final_results.bic:.2f}")

# -----------------------------------------------------------------------------
#                                  Пункт 5.3
# -----------------------------------------------------------------------------
gender_mapping = {1: 'м', 2: 'ж'}
spec_mapping = {
    1: 'Не_использую',
    2: 'Крайне_редко',
    3: 'Ежедневно',
    4: 'Постоянно'
}

df_group1['Пол_текст'] = df_group1['Пол'].map(gender_mapping)
df_group1['Спец_текст'] = df_group1['Специализация'].map(spec_mapping)

dummies_gender = pd.get_dummies(df_group1['Пол_текст'], drop_first=True)
dummies_spec = pd.get_dummies(df_group1['Спец_текст'], drop_first=True)

data_final_m3 = pd.concat([df_group1[['Возраст', 'Стаж', 'Доход', 'Страницы']], dummies_gender, dummies_spec], axis=1)

data_final_m3 = data_final_m3.apply(pd.to_numeric, errors='coerce')
bool_cols = data_final_m3.select_dtypes(include=['bool']).columns
data_final_m3[bool_cols] = data_final_m3[bool_cols].astype('int64')

X3 = data_final_m3  
X3_with_const = sm.add_constant(X3)

print("Матрица признаков X для Модели 3 с константой (Первые 5 строк):")
print(X3_with_const.head())

model3 = sm.OLS(y, X3_with_const)
results3 = model3.fit()

print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 3 =============================")
print(results3.summary())

predictions3 = results3.predict(X3_with_const)
residuals3 = y - predictions3

min_resid_m3 = residuals3.min()
max_resid_m3 = residuals3.max()
sd_resid_m3 = residuals3.std(ddof=1) 
me_m3 = residuals3.mean() 
mpe_m3 = (residuals3 / y).mean() * 100 

mae_m3 = np.mean(np.abs(residuals3))
rmse_m3 = np.sqrt(np.mean(residuals3**2))
r2_m3 = results3.rsquared
mape_m3 = np.mean(np.abs((y - predictions3) / y)) * 100

print("\nХарактеристики точности для Модели 3 (в Таблицу 1):")
print(f"ME (средняя ошибка): {me_m3:.2f}")
print(f"Min (минимальный остаток): {min_resid_m3:.2f}")
print(f"Max (максимальный остаток): {max_resid_m3:.2f}")
print(f"MAE (средняя абсолютная ошибка): {mae_m3:.2f}")
print(f"SD (стандартное отклонение остатков): {sd_resid_m3:.2f}")
print(f"MPE (средняя процентная ошибка): {mpe_m3:.2f}%")
print(f"MAPE (ср. абс. % ошибка): {mape_m3:.2f}%")
print(f"MSE (среднеквадратическая ошибка): {np.mean(residuals3**2):.2f}")
print(f"RMSE (корень из СКО): {rmse_m3:.2f}")
print(f"D (коэффициент детерминации): {r2_m3:.4f}")
print(f"AIC (критерий Акаике): {results3.aic:.2f}")
print(f"BIC (критерий Шварца): {results3.bic:.2f}")

# -----------------------------------------------------------------------------
#                                  Пункт 5.4 
# -----------------------------------------------------------------------------

features_final = ['Возраст', 'Доход', 'Страницы']

# --- СТРОИМ МОДЕЛЬ 4 (ГРУППА 1) ---
X4 = df_group1[features_final]
y4 = df_group1['Активность']
X4_with_const = sm.add_constant(X4)

model4 = sm.OLS(y4, X4_with_const)
results4 = model4.fit()

print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 4 =============================")
print(results4.summary())

# Сохраняем предсказания и остатки Модели 4
df_group1['Предсказания_М4'] = results4.predict(X4_with_const)
df_group1['Остатки_М4'] = results4.resid

# --- СТРОИМ МОДЕЛЬ 5 (ГРУППА 2) ---
# Фильтруем исходный датасет для второй группы
df_group2 = data[data['Группа'] == 2].copy()

X5 = df_group2[features_final]
y5 = df_group2['Активность']
X5_with_const = sm.add_constant(X5)

model5 = sm.OLS(y5, X5_with_const)
results5 = model5.fit()

print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 5 =============================")
print(results5.summary())

df_group2['Предсказания_М5'] = results5.predict(X5_with_const)
df_group2['Остатки_М5'] = results5.resid


# --- ГРАФИЧЕСКИЙ АНАЛИЗ ОСТАТКОВ ДЛЯ МОДЕЛЕЙ 4 И 5 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(df_group1['Предсказания_М4'], df_group1['Остатки_М4'], color='#1f77b4', alpha=0.7)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Предсказанные значения')
axes[0].set_ylabel('Остатки')
axes[0].set_title('График остатков Модели 4 (Группа 1)')
axes[0].grid(True, linestyle='--', alpha=0.3)

axes[1].scatter(df_group2['Предсказания_М5'], df_group2['Остатки_М5'], color='#ff7f0e', alpha=0.7)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Предсказанные значения')
axes[1].set_ylabel('Остатки')
axes[1].set_title('График остатков Модели 5 (Группа 2)')
axes[1].grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Гистограммы остатков для Модели 4 и 5
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df_group1['Остатки_М4'], kde=True, ax=axes[0], color='#1f77b4')
axes[0].set_title('Гистограмма остатков Модели 4')
axes[0].set_xlabel('Остатки')

sns.histplot(df_group2['Остатки_М5'], kde=True, ax=axes[1], color='#ff7f0e')
axes[1].set_title('Гистограмма остатков Модели 5')
axes[1].set_xlabel('Остатки')
plt.tight_layout()
plt.show()

# 3. Графики Q-Q для проверки нормальности Модели 4 и 5
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
stats.probplot(df_group1['Остатки_М4'], dist="norm", plot=axes[0])
axes[0].set_title('График Q-Q Модели 4 (Группа 1)')
axes[0].set_xlabel('Теоретические квантили')
axes[0].set_ylabel('Квантили выборки')

stats.probplot(df_group2['Остатки_М5'], dist="norm", plot=axes[1])
axes[1].set_title('График Q-Q Модели 5 (Группа 2)')
axes[1].set_xlabel('Теоретические квантили')
axes[1].set_ylabel('Квантили выборки')
plt.tight_layout()
plt.show()

# 4. Графики зависимости реальных и модельных значений
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(df_group1['Предсказания_М4'], y4, color='blue', alpha=0.7, edgecolors='w', s=60)
min_val4 = min(df_group1['Предсказания_М4'].min(), y4.min())
max_val4 = max(df_group1['Предсказания_М4'].max(), y4.max())
axes[0].plot([min_val4, max_val4], [min_val4, max_val4], 'r--', linewidth=2, label='Идеальная линия')
axes[0].set_xlabel('Предсказанные значения активности')
axes[0].set_ylabel('Реальные значения активности')
axes[0].set_title('Реальные vs Модельные (Модель 4)')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].scatter(df_group2['Предсказания_М5'], y5, color='orange', alpha=0.7, edgecolors='w', s=60)
min_val5 = min(df_group2['Предсказания_М5'].min(), y5.min())
max_val5 = max(df_group2['Предсказания_М5'].max(), y5.max())
axes[1].plot([min_val5, max_val5], [min_val5, max_val5], 'r--', linewidth=2, label='Идеальная линия')
axes[1].set_xlabel('Предсказанные значения активности')
axes[1].set_ylabel('Реальные значения активности')
axes[1].set_title('Реальные vs Модельные (Модель 5)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# --- РАСЧЕТ ХАРАКТЕРИСТИК ТОЧНОСТИ ДЛЯ ТАБЛИЦЫ 1 ---

def get_metrics_dict(residuals, predictions, y_actual, fitted_model):
    return {
        'ME': residuals.mean(),
        'Min': residuals.min(),
        'Max': residuals.max(),
        'MAE': np.mean(np.abs(residuals)),
        'SD': residuals.std(ddof=1),
        'MPE': np.mean(residuals / y_actual) * 100,
        'MAPE': np.mean(np.abs(residuals / y_actual)) * 100,
        'MSE': np.mean(residuals**2),
        'RMSE': np.sqrt(np.mean(residuals**2)),
        'D': fitted_model.rsquared,
        'AIC': fitted_model.aic,
        'BIC': fitted_model.bic
    }

metrics_m4 = get_metrics_dict(df_group1['Остатки_М4'], df_group1['Предсказания_М4'], y4, results4)
metrics_m5 = get_metrics_dict(df_group2['Остатки_М5'], df_group2['Предсказания_М5'], y5, results5)

print("\nХарактеристики точности для Модели 4 (в Таблицу 1):")
for metric, val in metrics_m4.items():
    print(f"{metric}: {val:.4f}")

print("\nХарактеристики точности для Модели 5 (в Таблицу 1):")
for metric, val in metrics_m5.items():
    print(f"{metric}: {val:.4f}")
