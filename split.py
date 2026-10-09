import argparse
import numpy as np
import pandas as pd


def split_dataset(data_path, train_out, valid_out, test_ratio=0.2, seed=42):
    np.random.seed(seed)

    df = pd.read_csv(data_path, header=None)

    if df.shape[1] != 32:
        raise ValueError("В файле должно быть 32 столбца")

    y_raw = df.iloc[:, 1].values

    if not np.all(np.isin(y_raw, ["B", "M"])):
        raise ValueError("В столбце диагноза должны быть только B и M")

    y = np.where(y_raw == "M", 1, 0)

    idx_B = np.where(y == 0)[0]
    idx_M = np.where(y == 1)[0]

    np.random.shuffle(idx_B)
    np.random.shuffle(idx_M)

    split_B = int(len(idx_B) * (1 - test_ratio))
    split_M = int(len(idx_M) * (1 - test_ratio))

    train_idx = np.concatenate([
        idx_B[:split_B],
        idx_M[:split_M]
    ])

    valid_idx = np.concatenate([
        idx_B[split_B:],
        idx_M[split_M:]
    ])

    np.random.shuffle(train_idx)
    np.random.shuffle(valid_idx)

    df.iloc[train_idx].to_csv(
        train_out,
        index=False,
        header=False
    )

    df.iloc[valid_idx].to_csv(
        valid_out,
        index=False,
        header=False
    )

    print("Разделение данных завершено")
    print("Всего:", len(df))
    print("Обучение:", len(train_idx))
    print("Проверка:", len(valid_idx))
    print("Файл обучения:", train_out)
    print("Файл проверки:", valid_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data.csv"
    )

    parser.add_argument(
        "--train_out",
        default="data_training.csv"
    )

    parser.add_argument(
        "--valid_out",
        default="data_validation.csv"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    split_dataset(
        args.dataset,
        args.train_out,
        args.valid_out,
        seed=args.seed
    )