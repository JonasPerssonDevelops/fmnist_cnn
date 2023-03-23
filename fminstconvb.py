#!/usr/bin/env python
# coding: utf-8
# Portland State Winter 2023
# CS441/541 Group Project - Team Shasta - 2 Layer Deep Learning Fashion MINST Image Classification

import tensorflow as tflow
import numpy as np
import matplotlib.pyplot as plt

def main():
    fashion_mnist = tflow.keras.datasets.fashion_mnist
    (training_images, training_labels), (
        test_images,
        test_labels,
    ) = fashion_mnist.load_data()

    fashion_enum = [
        "Tshirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    neuralnet = tflow.keras.Sequential(
        [
            tflow.keras.layers.Flatten(input_shape=(28, 28)),
            tflow.keras.layers.Dense(128, activation="relu"),
            tflow.keras.layers.Dense(10),
        ]
    )
    neuralnet.compile(
        optimizer="adam",
        loss=tflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    neuralnet.fit(training_images, training_labels, epochs=10)

    test_loss, test_accuracy = neuralnet.evaluate(test_images, test_labels, verbose=2)

    print("Accuracy of the model fit:", str(test_accuracy)[:4])
    print("\nAn ounce of prevention is worth a pound of cure. -Benjamin Franklin")

if __name__ == "__main__":
    main()
