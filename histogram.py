import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv("data.csv", header=None)
print("Общая информация о датасете:")

print(f"Размерность: {df.shape[0]} строк, {df.shape[1]} столбцов")

print(
    f"Количество пропусков: "
    f"{df.isnull().sum().sum()}"
)

print("\nРаспределение целевого класса (diagnosis):")

class_counts = df[1].value_counts()

print(f"Количество B: {class_counts.get('B', 0)}")
print(f"Количество M: {class_counts.get('M', 0)}")

print("\nПроцентное соотношение:")

class_percent = df[1].value_counts(normalize=True) * 100

for class_name, percent in class_percent.items():
    print(f"{class_name}: {percent:.2f}%")

print("\nСтатистические характеристики:")

features = df.iloc[:, 2:]
features.columns = FEATURE_NAMES

stats = features.describe().T[
    ["count", "mean", "std", "min", "50%", "max"]
]

stats.columns = [
    "Количество",
    "Среднее",
    "Стд. откл.",
    "Мин.",
    "Медиана",
    "Макс."
]

print(stats)


fig, axes = plt.subplots(
    6,
    5,
    figsize=(15, 18)
)

for i in range(30):
    axes[i // 5, i % 5].hist(
        df[i + 2],
        bins=30
    )

    axes[i // 5, i % 5].set_title(
        FEATURE_NAMES[i]
    )

    axes[i // 5, i % 5].grid()


plt.tight_layout()

plt.savefig(
    "histogram_all.png"
)

plt.show()

fig, axes = plt.subplots(
    6,
    5,
    figsize=(15, 18)
)

for i in range(30):

    axes[i // 5, i % 5].hist(
        df[df[1] == "B"][i + 2],
        bins=30,
        alpha=0.5,
        label="B"
    )

    axes[i // 5, i % 5].hist(
        df[df[1] == "M"][i + 2],
        bins=30,
        alpha=0.5,
        label="M"
    )

    axes[i // 5, i % 5].set_title(
        FEATURE_NAMES[i]
    )

    axes[i // 5, i % 5].grid()


axes[0, 0].legend()


plt.tight_layout()

plt.savefig(
    "histogram_classes.png"
)

plt.show()


