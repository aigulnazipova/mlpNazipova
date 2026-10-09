import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(42)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
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


def one_hot(y, classes):
    result = np.zeros(
        (len(y), classes)
    )

    result[
        np.arange(len(y)),
        y
    ] = 1

    return result


class MLP:
    def __init__(self, layer_sizes):

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):

            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            limit = np.sqrt(6 / n_in)

            weights = np.random.uniform(
                -limit,
                limit,
                (n_in, n_out)
            )

            biases = np.zeros(
                (1, n_out)
            )

            self.weights.append(weights)
            self.biases.append(biases)

    def forward(self, X):

        activations = [X]

        for i in range(
            len(self.weights) - 1
        ):

            z = (
                activations[-1] @ self.weights[i]
                + self.biases[i]
            )

            a = sigmoid(z)

            activations.append(a)

        z = (
            activations[-1] @ self.weights[-1]
            + self.biases[-1]
        )

        a = softmax(z)

        activations.append(a)

        return activations

    def backward(
        self,
        activations,
        y,
        learning_rate
    ):

        m = len(y)

        gradients_w = [
            None
        ] * len(self.weights)

        gradients_b = [
            None
        ] * len(self.biases)

        dz = (
            activations[-1] - y
        ) / m

        for i in range(
            len(self.weights) - 1,
            -1,
            -1
        ):

            gradients_w[i] = (
                activations[i].T @ dz
            )

            gradients_b[i] = np.sum(
                dz,
                axis=0,
                keepdims=True
            )

            if i > 0:

                da = (
                    dz @ self.weights[i].T
                )

                dz = (
                    da
                    * activations[i]
                    * (1 - activations[i])
                )

        for i in range(
            len(self.weights)
        ):

            self.weights[i] -= (
                learning_rate
                * gradients_w[i]
            )

            self.biases[i] -= (
                learning_rate
                * gradients_b[i]
            )


def train():

    parser = argparse.ArgumentParser(
        description="Обучение многослойного персептрона"
    )

    parser.add_argument(
        "--dataset",
        default="data_training.csv"
    )

    parser.add_argument(
        "--valid_dataset",
        default="data_validation.csv"
    )

    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[24, 24]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=55
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.03
    )

    args = parser.parse_args()

    train_data = pd.read_csv(
        args.dataset,
        header=None
    )

    valid_data = pd.read_csv(
        args.valid_dataset,
        header=None
    )

    X_train = train_data.iloc[
        :, 2:
    ].values.astype(float)

    y_train = np.where(
        train_data.iloc[:, 1] == "M",
        1,
        0
    )

    X_valid = valid_data.iloc[
        :, 2:
    ].values.astype(float)

    y_valid = np.where(
        valid_data.iloc[:, 1] == "M",
        1,
        0
    )
    mean = X_train.mean(axis=0)

    std = X_train.std(axis=0)

    X_train = (
        X_train - mean
    ) / (std + 1e-8)

    X_valid = (
        X_valid - mean
    ) / (std + 1e-8)

    y_train = one_hot(
        y_train,
        2
    )

    y_valid = one_hot(
        y_valid,
        2
    )

    print(
        f"x_train shape : {X_train.shape}"
    )

    print(
        f"x_valid shape : {X_valid.shape}"
    )

    layer_sizes = (
        [X_train.shape[1]]
        + args.layer
        + [2]
    )

    print(
        f"Architecture : {layer_sizes}"
    )

    model = MLP(
        layer_sizes
    )
    train_losses = []
    valid_losses = []

    train_accs = []
    valid_accs = []

    for epoch in range(
        args.epochs
    ):
        indices = np.random.permutation(
            len(X_train)
        )

        X_shuffled = X_train[
            indices
        ]

        y_shuffled = y_train[
            indices
        ]

        for i in range(
            0,
            len(X_shuffled),
            args.batch_size
        ):

            X_batch = X_shuffled[
                i:i + args.batch_size
            ]

            y_batch = y_shuffled[
                i:i + args.batch_size
            ]

            activations = model.forward(
                X_batch
            )
            model.backward(
                activations,
                y_batch,
                args.learning_rate
            )

        train_pred = model.forward(
            X_train
        )[-1]

        valid_pred = model.forward(
            X_valid
        )[-1]

        train_loss = cross_entropy(
            y_train,
            train_pred
        )

        valid_loss = cross_entropy(
            y_valid,
            valid_pred
        )

        train_acc = np.mean(
            np.argmax(
                train_pred,
                axis=1
            )
            ==
            np.argmax(
                y_train,
                axis=1
            )
        )

        valid_acc = np.mean(
            np.argmax(
                valid_pred,
                axis=1
            )
            ==
            np.argmax(
                y_valid,
                axis=1
            )
        )

        train_losses.append(
            train_loss
        )

        valid_losses.append(
            valid_loss
        )

        train_accs.append(
            train_acc
        )

        valid_accs.append(
            valid_acc
        )

        print(
            f"epoch {epoch + 1:02d}/{args.epochs:02d} "
            f"- loss: {train_loss:.4f} "
            f"- val_loss: {valid_loss:.4f} "
            f"- acc: {train_acc:.4f} "
            f"- val_acc: {valid_acc:.4f}"
        )

    model_data = {
        "weights": model.weights,
        "biases": model.biases,
        "layer_sizes": layer_sizes,
        "mean": mean,
        "std": std
    }

    np.save(
        "saved_model.npy",
        model_data,
        allow_pickle=True
    )

    print(
        "\nМодель сохранена в "
        "'saved_model.npy'"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        train_losses,
        label="Training loss"
    )

    plt.plot(
        valid_losses,
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "loss_curve.png"
    )

    plt.show()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        train_accs,
        label="Training accuracy"
    )

    plt.plot(
        valid_accs,
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "accuracy_curve.png"
    )

    plt.show()


if __name__ == "__main__":
    train()