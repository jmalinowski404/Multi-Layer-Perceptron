import numpy as np

class Tester:
    def __init__(self, mlp):
        self.mlp = mlp

    def test(self, patterns, log_file="test_log.txt", is_classification=True):
        total_error = 0.0

        confusion_matrix = np.zeros((3, 3), dtype=int)
        correct_predictions = 0

        with open(log_file, "w", encoding="utf-8") as f:
            for p in patterns:
                inputs, expected = p
                output = self.mlp.forward(inputs)

                pattern_error = 0.5 * sum((e - o) ** 2 for e, o in zip(expected, output))
                output_errors = [(e - o) for e, o in zip(expected, output)]
                total_error += pattern_error

                f.write(f"Wejście: {inputs}\n")
                f.write(f"Oczekiwane: {expected}\n")
                f.write(f"Wyjście mlp: {[round(float(o), 3) for o in output]}\n")
                f.write(f"Błąd wzorca: {pattern_error}\n")
                f.write(f"Błędy na wyjściach: {[float(err) for err in output_errors]}\n")

                f.write("Szczegóły warstw (od wyjścia do wejścia):\n")
                for layer_idx in range(len(self.mlp.layers) - 1, -1, -1):
                    layer_name = "Wyjściowa" if layer_idx == len(self.mlp.layers) - 1 else f"Ukryta {layer_idx}"
                    f.write(f"  Warstwa: {layer_name}\n")
                    for n_idx, neuron in enumerate(self.mlp.layers[layer_idx].neurons):
                        f.write(
                            f"    Neuron {n_idx}: wyjście={float(neuron.output):.4f}, wagi={[round(float(w), 4) for w in neuron.weights]}\n")
                f.write("-" * 40 + "\n")

                if is_classification and len(expected) == 3:
                    predicted_class = np.argmax(output)
                    expected_class = np.argmax(expected)

                    confusion_matrix[expected_class][predicted_class] += 1
                    if predicted_class == expected_class:
                        correct_predictions += 1

            avg_error = total_error / len(patterns)
            f.write(f"\nŚredni błąd: {avg_error}\n")

            if is_classification and len(patterns[0][1]) == 3:
                f.write(f"\nPoprawnie sklasyfikowane: {correct_predictions} / {len(patterns)}\n")
                accuracy = (correct_predictions / len(patterns)) * 100
                f.write(f"Dokładność perceptronu: {accuracy}%\n")
                f.write("Macierz pomyłek:\n")
                f.write(str(confusion_matrix) + "\n\n")

                for i in range(3):
                    TP = confusion_matrix[i][i]
                    FP = sum(confusion_matrix[:, i]) - TP
                    FN = sum(confusion_matrix[i, :]) - TP

                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    f.write(f"Klasa {i}: Precision={precision:.2f}, Recall={recall:.2f}, F-measure={f_measure:.2f}\n")
                    print(f"Klasa {i} - P: {precision:.2f}, R: {recall:.2f}, F1: {f_measure:.2f}")

        print(f"Test zakończony. Średni błąd: {avg_error:.4f}. Wyniki zapisano w {log_file}")
        return avg_error