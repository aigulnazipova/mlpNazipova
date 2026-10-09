import argparse
import numpy as np
import pandas as pd


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True
    )


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(
        y_pred,
        1e-8,
        1 - 1e-8
    )

    return -np.mean(
        np.sum(
            y_true * np.log(y_pred),
            axis=1
        )
    )


def predict():

    parser = argparse.ArgumentParser(
        description="Предсказание с помощью обученной модели"
    )

    parser.add_argument(
        "--dataset",
        default="data_validation.csv"
    )

    parser.add_argument(
        "--model",
        default="saved_model.npy"
    )

    args = parser.parse_args()


    model_data = np.load(
        args.model,
        allow_pickle=True
    ).item()

    weights = model_data["weights"]
    biases = model_data["biases"]

    layer_sizes = model_data["layer_sizes"]

    mean = model_data["mean"]
    std = model_data["std"]

    print(
        f"Architecture : {layer_sizes}"
    )


    data = pd.read_csv(
        args.dataset,
        header=None
    )

    X = data.iloc[
        :, 2:
    ].values.astype(float)

    y = np.where(
        data.iloc[:, 1] == "M",
        1,
        0
    )

    X = (
        X - mean
    ) / (std + 1e-8)


    y_onehot = np.zeros(
        (len(y), 2)
    )

    y_onehot[
        np.arange(len(y)),
        y
    ] = 1

    activations = X

    for i in range(
        len(weights) - 1
    ):

        z = (
            activations @ weights[i]
            + biases[i]
        )

        activations = sigmoid(z)

    z = (
        activations @ weights[-1]
        + biases[-1]
    )

    predictions_prob = softmax(z)

    predictions = np.argmax(
        predictions_prob,
        axis=1
    )

    loss = cross_entropy(
        y_onehot,
        predictions_prob
    )

    accuracy = np.mean(
        predictions == y
    )

    correct = np.sum(
        predictions == y
    )

    incorrect = len(y) - correct

    print("Результаты предсказания:\n")

    print(
        f"Размер набора данных: {len(y)}"
    )

    print(
        f"Правильных предсказаний: "
        f"{correct}"
    )

    print(
        f"Ошибочных предсказаний: "
        f"{incorrect}"
    )

    print(
        f"Cross-Entropy Loss: "
        f"{loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )



if __name__ == "__main__":
    predict()