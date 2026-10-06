# Импорт необходимых библиотек с псевдонимом
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Импорт модуля из библиотеки
from scipy import stats
from scipy.stats import pearsonr

data = pd.read_csv("v_3.csv", sep=';', encoding="cp1251")
# Преобразуем данные в таблицу
data = pd.DataFrame(data)
# Выводим первые 5 строк таблицы, чтобы увидеть данные
print("Первые 5 строк данных:")
print(data.head())
print("\nИнформация о структуре данных:")
print(data.info())

new_column_names = ['№п/п', 'Группа', 'Пол', 'Возраст','Стаж','Доход','Специализация', 'Страницы', 'Активность'] 
# Присваиваем новый список имен 
data.columns = new_column_names 
print(data.info())

# Список категориальных переменных для преобразования
categorical_columns = [
    "Группа",
    "Пол",
    "Специализация"
]

# Преобразуем каждый столбец в тип 'category'
for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].astype("category")
        print(f"Столбец '{col}' преобразован в категориальный.")

# Проверяем результат
print("\nТипы данных после преобразования:")
print(data.dtypes)
print(data.info())

# -----------------------------------------------------------------------------
#                                  Пункт 5
# -----------------------------------------------------------------------------

# Функция для определения статистической значимости коэффициента корреляции:
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

