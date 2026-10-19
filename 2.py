import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
import statsmodels.api as sm

# -----------------------------------------------------------------------------
#                                Пункты 1 - 3
# -----------------------------------------------------------------------------

data = pd.read_csv("v_3.csv", sep=";", encoding="cp1251")
data = pd.DataFrame(data)
print("Первые 5 строк данных:")
print(data.head())

print("\nИнформация о структуре данных:")
print(data.info())

new_column_names = [
    "№п/п",
    "Группа",
    "Пол",
    "Возраст",
    "Доход",
    "Стаж",
    "Специализация",
    "Страницы",
    "Активность",
]
data.columns = new_column_names
print("\nНазвания столбцов изменены:")
print(data.info())

categorical_columns = ["Группа", "Пол", "Специализация"]

for col in categorical_columns:
  if col in data.columns:
    data[col] = data[col].astype("category")
    print(f"Столбец '{col}' преобразован в категориальный.")

print("\nТипы данных после преобразования:")
print(data.info())


# -----------------------------------------------------------------------------
#                          Пункт 4. Матричный график
# -----------------------------------------------------------------------------


def signif_stars(p):
  if p < 0.001:
    return "***"
  elif p < 0.01:
    return "**"
  elif p < 0.05:
    return "*"
  else:
    return ""


def ggpairs_like(df, features, hue="Группа"):
  n = len(features)
  classes = sorted(df[hue].unique())

  colors = ["#1f77b4a0", "#ff7f0ea0", "#2ca02ca0", "#d62728a0"]
  color_map = dict(zip(classes, colors[: len(classes)]))

  fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  for i in range(n):
    for j in range(n):
      ax = axes[i, j]
      xvar, yvar = features[j], features[i]

      if i < n - 1:
        ax.tick_params(bottom=False, labelbottom=False)
      else:
        ax.tick_params(bottom=True, labelbottom=True)

      if j > 0:
        ax.tick_params(left=False, labelleft=False)
      else:
        ax.tick_params(left=True, labelleft=True)
        ax.tick_params(axis="y", labelsize=8)

      if i == j:
        if pd.api.types.is_numeric_dtype(df[xvar]):
          for cls in classes:
            cls_data = df[df[hue] == cls][xvar]
            sns.kdeplot(
                data=cls_data,
                fill=True,
                alpha=0.6,
                color=color_map[cls],
                ax=ax,
                label=cls,
            )
          ax.legend().remove()
          ax.grid(True, linestyle="--", alpha=0.3)
          ax.set_ylabel("")
        else:
          sns.countplot(data=df, x=xvar, hue=hue, ax=ax, palette=color_map)
          ax.grid(True, linestyle="--", alpha=0.3, axis="y")
          ax.set_ylabel("")

      elif i > j:
        if pd.api.types.is_numeric_dtype(
            df[xvar]
        ) and pd.api.types.is_numeric_dtype(df[yvar]):
          for cls in classes:
            cls_data = df[df[hue] == cls]
            ax.scatter(
                cls_data[xvar],
                cls_data[yvar],
                alpha=0.6,
                s=12,
                color=color_map[cls],
                label=cls,
            )
          ax.legend().remove()
          ax.grid(True, linestyle="--", alpha=0.3)
        else:
          ax.axis("off")
      else:
        if pd.api.types.is_numeric_dtype(
            df[xvar]
        ) and pd.api.types.is_numeric_dtype(df[yvar]):
          valid_data = df[[xvar, yvar]].dropna()
          if len(valid_data) > 2:
            r, p = pearsonr(valid_data[xvar], valid_data[yvar])

            y_pos = 0.85
            ax.text(
                0.5,
                y_pos,
                f"Corr: {r:.3f}{signif_stars(p)}",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", alpha=0.9
                ),
            )

            y_pos -= 0.20
            for cls_idx, cls in enumerate(classes):
              sub = df[df[hue] == cls]
              valid_sub = sub[[xvar, yvar]].dropna()
              if len(valid_sub) > 2:
                r_c, p_c = pearsonr(valid_sub[xvar], valid_sub[yvar])
                ax.text(
                    0.5,
                    y_pos,
                    f"{cls}: {r_c:.3f}{signif_stars(p_c)}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color_map[cls],
                    transform=ax.transAxes,
                    bbox=dict(
                        boxstyle="round,pad=0.2", facecolor="white", alpha=0.9
                    ),
                )
              y_pos -= 0.18
          else:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
        ax.axis("off")

      if i == 0 and j != 0:
        if pd.api.types.is_numeric_dtype(df[xvar]):
          boxplot_data = [df[df[hue] == cls][xvar].dropna() for cls in classes]
          bp = ax.boxplot(boxplot_data, patch_artist=True)

          for patch, cls in zip(bp["boxes"], classes):
            patch.set_facecolor(color_map[cls])
            patch.set_alpha(0.7)

          ax.set_xticks(range(1, len(classes) + 1))
          ax.set_xticklabels(classes, fontsize=8)
          ax.grid(True, linestyle="--", alpha=0.3, axis="y")
          ax.set_ylabel("")

      if j == 0 and i != 0:
        if pd.api.types.is_numeric_dtype(df[yvar]):
          ax.clear()
          n_classes = len(classes)
          y_min, y_max = df[yvar].min(), df[yvar].max()
          bins = np.histogram_bin_edges(df[yvar], bins="auto")

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
                rect = plt.Rectangle(
                    (x_left, y_bottom),
                    height,
                    y_top - y_bottom,
                    facecolor=color_map[cls],
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

          ax.set_xlim(0, 1)
          ax.set_ylim(y_min, y_max)
          ax.set_ylabel("")
          ax.set_xticks([])
          ax.set_xticklabels([])
          ax.grid(True, linestyle="--", alpha=0.3, axis="y")
          ax.yaxis.set_major_locator(plt.MaxNLocator(4))
          ax.tick_params(axis="y", labelsize=8)

      ax.grid(True, linestyle="--", alpha=0.3)

  for j in range(n):
    axes[0, j].set_title(
        features[j].upper(), fontsize=11, fontweight="bold", pad=15
    )

  for i in range(n):
    axes[i, n - 1].text(
        1.15,
        0.5,
        features[i].upper(),
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        rotation=270,
        transform=axes[i, n - 1].transAxes,
    )

  axes[n - 1, n - 1].set_xlabel("")
  plt.tight_layout()
  plt.show()


ggpairs_like(data, ["Группа", "Возраст", "Доход", "Стаж", "Страницы", "Активность"])
ggpairs_like(data, ["Пол", "Возраст", "Доход", "Стаж", "Страницы", "Активность"])
ggpairs_like(data,["Специализация", "Возраст", "Доход", "Стаж", "Страницы", "Активность"],)


# Вспомогательные функции для расчета метрик и печати уравнения
def get_metrics_dict(residuals, predictions, y_actual, fitted_model):
  return {
      "ME": residuals.mean(),
      "Min": residuals.min(),
      "Max": residuals.max(),
      "MAE": np.mean(np.abs(residuals)),
      "SD": residuals.std(ddof=1),
      "MPE": np.mean(residuals / y_actual) * 100,
      "MAPE": np.mean(np.abs(residuals / y_actual)) * 100,
      "MSE": np.mean(residuals**2),
      "RMSE": np.sqrt(np.mean(residuals**2)),
      "D": fitted_model.rsquared,
      "AIC": fitted_model.aic,
      "BIC": fitted_model.bic,
  }


def print_regression_equation(fitted_model, target_name="Активность"):
  params = fitted_model.params
  eq = f"{target_name} = {params['const']:.4f}"
  for col_name, val in params.items():
    if col_name != "const":
      sign = "+" if val >= 0 else "-"
      eq += f" {sign} {abs(val):.4f} * {col_name}"
  print("\nУравнение регрессии:")
  print(eq)


# -----------------------------------------------------------------------------
#                                  Пункт 5.1
# -----------------------------------------------------------------------------

df_group1 = data[data["Группа"] == 1].copy()

X1 = df_group1[["Возраст", "Доход", "Стаж", "Страницы"]]
y1 = df_group1["Активность"]
X1_with_const = sm.add_constant(X1)

model1 = sm.OLS(y1, X1_with_const)
results1 = model1.fit()

print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 1 =============================")
print(results1.summary())
print_regression_equation(results1)

residuals1 = results1.resid
df_group1["Предсказания"] = results1.predict(X1_with_const)
df_group1["Остатки"] = residuals1

# Графики для Модели 1
plt.figure(figsize=(8, 6))
plt.scatter(df_group1["Предсказания"], df_group1["Остатки"])
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Предсказанные значения")
plt.ylabel("Остатки")
plt.title("График остатков для проверки гомоскедастичности Модели 1")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(residuals1, kde=True, ax=ax[0])
ax[0].set_title("Гистограмма остатков Модели 1")
ax[0].set_xlabel("Остатки")

stats.probplot(residuals1, dist="norm", plot=ax[1])
ax[1].set_title("График Q-Q для проверки нормальности Модели 1")
ax[1].set_xlabel("Теоретические квантили")
ax[1].set_ylabel("Квантили выборки")
plt.tight_layout()
plt.show()

predictions1 = results1.predict(X1_with_const)
plt.figure(figsize=(8, 6))
plt.scatter(predictions1, y1, color="blue", alpha=0.7, edgecolors="w", s=80)
min_val = min(min(predictions1), min(y1))
max_val = max(max(predictions1), max(y1))
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Идеальная линия",
)
plt.xlabel("Предсказанные значения активности", fontsize=12)
plt.ylabel("Реальные значения активности", fontsize=12)
plt.title("График: Реальные и Предсказанные значения для Модели 1", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

metrics1 = get_metrics_dict(df_group1["Остатки"], df_group1["Предсказания"], y1, results1)

# Пошаговый матричный расчет "вручную" для Модели 1 (п. 5.1.7)
X_mat = X1_with_const.values
y_vec = y1.values
n_obs, m_params = X_mat.shape
k_preds = m_params - 1

XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
beta_manual = XtX_inv @ (X_mat.T @ y_vec)
y_hat_manual = X_mat @ beta_manual
e_manual = y_vec - y_hat_manual

RSS = np.sum(e_manual**2)
TSS = np.sum((y_vec - np.mean(y_vec)) ** 2)
R2_manual = 1 - RSS / TSS
s2 = RSS / (n_obs - m_params)
se_manual = np.sqrt(np.diag(XtX_inv) * s2)
t_manual = beta_manual / se_manual
F_manual = (R2_manual / k_preds) / ((1 - R2_manual) / (n_obs - m_params))

print("\n--- Пошаговая проверка МНК (Модель 1) ---")
for name, b, se, t_val in zip(
    X1_with_const.columns, beta_manual, se_manual, t_manual
):
  print(f"{name:10s}: beta = {b:8.4f}, SE = {se:8.4f}, t = {t_val:8.4f}")
print(f"R2 (вручную) = {R2_manual:.4f}, F-статистика (вручную) = {F_manual:.4f}")


# -----------------------------------------------------------------------------
#                                  Пункт 5.2
# -----------------------------------------------------------------------------

X2_with_const = sm.add_constant(X1.copy())
y2 = y1
SL = 0.05

num_vars = len(X1.columns)
for i in range(num_vars):
  results2 = sm.OLS(y2, X2_with_const).fit()
  max_p_value = max(results2.pvalues[1:])  
  if max_p_value > SL:
    j = np.argmax(results2.pvalues[1:]) + 1
    dropped_col = X2_with_const.columns[j]
    print(f"\nПошаговый отбор: исключаем '{dropped_col}' (p = {max_p_value:.4f})")
    X2_with_const = X2_with_const.drop(dropped_col, axis=1)
  else:
    break

print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 2 =============================")
final_results2 = sm.OLS(y2, X2_with_const).fit()
print(final_results2.summary())
print_regression_equation(final_results2)

predictions2 = final_results2.predict(X2_with_const)
residuals2 = y2 - predictions2
metrics2 = get_metrics_dict(residuals2, predictions2, y2, final_results2)


# -----------------------------------------------------------------------------
#                                  Пункт 5.3
# -----------------------------------------------------------------------------

y3 = y1
gender_mapping = {1: "м", 2: "ж"}
spec_mapping = {
    1: "Не_использую",
    2: "Крайне_редко",
    3: "Ежедневно",
    4: "Постоянно",
}

df_group1["Пол_текст"] = df_group1["Пол"].cat.rename_categories(gender_mapping)
df_group1["Спец_текст"] = df_group1["Специализация"].cat.rename_categories(spec_mapping) 

dummies_gender1 = pd.get_dummies(df_group1["Пол_текст"], drop_first=True, dtype="int64")
dummies_spec1 = pd.get_dummies(df_group1["Спец_текст"], drop_first=True, dtype="int64")

X3 = pd.concat([df_group1[["Возраст", "Доход", "Стаж", "Страницы"]],dummies_gender1,dummies_spec1,],axis=1,)
X3_with_const = sm.add_constant(X3)

print("\nМатрица признаков X для Модели 3 с константой (первые 5 строк):")
print(X3_with_const.head())

final_results3 = sm.OLS(y3, X3_with_const).fit()
print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 3 =============================")
print(final_results3.summary())
print_regression_equation(final_results3)

predictions3 = final_results3.predict(X3_with_const)
residuals3 = y3 - predictions3
metrics3 = get_metrics_dict(residuals3, predictions3, y3, final_results3)


# -----------------------------------------------------------------------------
#                                  Пункт 5.4
# -----------------------------------------------------------------------------

quant_features_final = ["Возраст", "Доход", "Страницы"]

# Модель 4 (Группа 1)
X4 = pd.concat([df_group1[quant_features_final], dummies_spec1], axis=1)
y4 = df_group1["Активность"]
X4_with_const = sm.add_constant(X4)

results4 = sm.OLS(y4, X4_with_const).fit()
print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 4 (Группа 1) =============================")
print(results4.summary())
print_regression_equation(results4)

df_group1["Предсказания4"] = results4.predict(X4_with_const)
df_group1["Остатки4"] = results4.resid

# Модель 5 (Группа 2)
df_group2 = data[data["Группа"] == 2].copy()
df_group2["Спец_текст"] = df_group2["Специализация"].cat.rename_categories(
    spec_mapping
)
dummies_spec2 = pd.get_dummies(
    df_group2["Спец_текст"], drop_first=True, dtype="int64"
)

X5 = pd.concat([df_group2[quant_features_final], dummies_spec2], axis=1)
y5 = df_group2["Активность"]
X5_with_const = sm.add_constant(X5)

results5 = sm.OLS(y5, X5_with_const).fit()
print("\n============================ СВОДКА ДЛЯ МОДЕЛИ 5 (Группа 2) =============================")
print(results5.summary())
print_regression_equation(results5)

df_group2["Предсказания5"] = results5.predict(X5_with_const)
df_group2["Остатки5"] = results5.resid

# Графики остатков для Моделей 4 и 5
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(
    df_group1["Предсказания4"],
    df_group1["Остатки4"],
    color="#1f77b4",
    alpha=0.7,
)
axes[0].axhline(y=0, color="r", linestyle="--")
axes[0].set_xlabel("Предсказанные значения")
axes[0].set_ylabel("Остатки")
axes[0].set_title("График остатков Модели 4 (Группа 1)")
axes[0].grid(True, linestyle="--", alpha=0.3)

axes[1].scatter(
    df_group2["Предсказания5"],
    df_group2["Остатки5"],
    color="#ff7f0e",
    alpha=0.7,
)
axes[1].axhline(y=0, color="r", linestyle="--")
axes[1].set_xlabel("Предсказанные значения")
axes[1].set_ylabel("Остатки")
axes[1].set_title("График остатков Модели 5 (Группа 2)")
axes[1].grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# Гистограммы остатков для Моделей 4 и 5
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df_group1["Остатки4"], kde=True, ax=axes[0], color="#1f77b4")
axes[0].set_title("Гистограмма остатков Модели 4")
axes[0].set_xlabel("Остатки")

sns.histplot(df_group2["Остатки5"], kde=True, ax=axes[1], color="#ff7f0e")
axes[1].set_title("Гистограмма остатков Модели 5")
axes[1].set_xlabel("Остатки")
plt.tight_layout()
plt.show()

# Графики Q-Q для проверки нормальности Моделей 4 и 5
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
stats.probplot(df_group1["Остатки4"], dist="norm", plot=axes[0])
axes[0].set_title("График Q-Q Модели 4 (Группа 1)")
axes[0].set_xlabel("Теоретические квантили")
axes[0].set_ylabel("Квантили выборки")

stats.probplot(df_group2["Остатки5"], dist="norm", plot=axes[1])
axes[1].set_title("График Q-Q Модели 5 (Группа 2)")
axes[1].set_xlabel("Теоретические квантили")
axes[1].set_ylabel("Квантили выборки")
plt.tight_layout()
plt.show()

# Графики зависимости реальных и модельных значений для Моделей 4 и 5
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(
    df_group1["Предсказания4"],
    y4,
    color="blue",
    alpha=0.7,
    edgecolors="w",
    s=60,
)
min_val4 = min(df_group1["Предсказания4"].min(), y4.min())
max_val4 = max(df_group1["Предсказания4"].max(), y4.max())
axes[0].plot(
    [min_val4, max_val4],
    [min_val4, max_val4],
    "r--",
    linewidth=2,
    label="Идеальная линия",
)
axes[0].set_xlabel("Предсказанные значения активности")
axes[0].set_ylabel("Реальные значения активности")
axes[0].set_title("Реальные vs Модельные (Модель 4)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].scatter(
    df_group2["Предсказания5"],
    y5,
    color="orange",
    alpha=0.7,
    edgecolors="w",
    s=60,
)
min_val5 = min(df_group2["Предсказания5"].min(), y5.min())
max_val5 = max(df_group2["Предсказания5"].max(), y5.max())
axes[1].plot(
    [min_val5, max_val5],
    [min_val5, max_val5],
    "r--",
    linewidth=2,
    label="Идеальная линия",
)
axes[1].set_xlabel("Предсказанные значения активности")
axes[1].set_ylabel("Реальные значения активности")
axes[1].set_title("Реальные vs Модельные (Модель 5)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

metrics4 = get_metrics_dict(
    df_group1["Остатки4"], df_group1["Предсказания4"], y4, results4
)
metrics5 = get_metrics_dict(
    df_group2["Остатки5"], df_group2["Предсказания5"], y5, results5
)


# -----------------------------------------------------------------------------
#                     Сводная Таблица 1 и Прогноз (Пункт 8)
# -----------------------------------------------------------------------------

table1_df = pd.DataFrame({
    "Модель 1": metrics1,
    "Модель 2": metrics2,
    "Модель 3": metrics3,
    "Модель 4": metrics4,
    "Модель 5": metrics5,
})

print("\n============================ ТАБЛИЦА 1. ХАРАКТЕРИСТИКИ ТОЧНОСТИ =============================")
print(table1_df.round(4))


new_object = pd.DataFrame({
    "const": [1.0],
    "Возраст": [27],
    "Доход": [65.0],
    "Страницы": [1200.0],
    "Крайне_редко": [0],
    "Ежедневно": [1],
    "Постоянно": [0],
})

pred_group1 = results4.predict(new_object).iloc[0]
pred_group2 = results5.predict(new_object).iloc[0]

print("\n--- Прогноз степени активности для нового объекта ---")
print(new_object.drop(columns=["const"]))
print(f"Прогноз по Модели 4 (Группа 1): {pred_group1:.2f} балла")
print(f"Прогноз по Модели 5 (Группа 2): {pred_group2:.2f} балла")
