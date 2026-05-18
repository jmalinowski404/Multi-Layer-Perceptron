import re
import matplotlib.pyplot as plt


def generuj_wykres_z_logu(nazwa_pliku_wejsciowego, nazwa_pliku_wyjsciowego='wykres_bledu.png'):
    """
    Funkcja odczytuje logi z pliku tekstowego, parsuje epoki oraz błędy,
    a następnie generuje wykres i eksportuje go do pliku PNG.
    """
    try:
        with open(nazwa_pliku_wejsciowego, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{nazwa_pliku_wejsciowego}'.")
        return

    cleaned_content = re.sub(r'\\', '', content)

    matches = re.findall(r'Epoka\s+(\d+):\s*([0-9.]+)', cleaned_content)

    epochs = []
    errors = []

    for ep, err in matches:
        epochs.append(int(ep))
        errors.append(float(err))

    if not epochs:
        print("Nie udało się znaleźć poprawnych danych w pliku.")
        return

    sorted_data = sorted(zip(epochs, errors))
    epochs, errors = zip(*sorted_data)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, errors, linestyle='-', color='blue', alpha=0.8, linewidth=1.5)

    plt.title('Zależność błędu od epoki', fontsize=14, fontweight='bold')
    plt.xlabel('Epoka', fontsize=12)
    plt.xticks(range(0, max(epochs) + 1, 500), rotation=45)
    plt.ylabel('Błąd', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(nazwa_pliku_wyjsciowego, dpi=300)
    print(f"Pomyślnie wygenerowano wykres i zapisano jako: '{nazwa_pliku_wyjsciowego}'.")

    plt.close()


import re
import numpy as np
import matplotlib.pyplot as plt


def rysuj_macierz_pomylek(nazwa_pliku, plik_wyjsciowy='macierz_pomylek.png'):
    try:
        with open(nazwa_pliku, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{nazwa_pliku}'.")
        return

    acc_match = re.search(r'Dokładność perceptronu:\s*([0-9.]+)%', content)
    accuracy = acc_match.group(1) if acc_match else "Nieznana"

    matrix_match = re.search(r'Macierz pomyłek:\n(\[\[.*?\]\])', content, re.DOTALL)
    if not matrix_match:
        print("Nie znaleziono macierzy pomyłek w pliku.")
        return

    clean_str = matrix_match.group(1).replace('[', '').replace(']', '').strip()
    wiersze = clean_str.split('\n')

    macierz = []
    for w in wiersze:
        macierz.append([int(x) for x in w.split()])

    macierz = np.array(macierz)

    fig, ax = plt.subplots(figsize=(7, 6))

    cax = ax.matshow(macierz, cmap='Blues', alpha=0.8)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    for i in range(macierz.shape[0]):
        for j in range(macierz.shape[1]):
            kolor_tekstu = "white" if macierz[i, j] > (macierz.max() / 2) else "black"
            ax.text(j, i, str(macierz[i, j]), va='center', ha='center',
                    color=kolor_tekstu, fontsize=14, fontweight='bold')

    klasy = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    ax.set_xticks(range(len(klasy)))
    ax.set_yticks(range(len(klasy)))
    ax.set_xticklabels(klasy)
    ax.set_yticklabels(klasy)

    ax.xaxis.set_ticks_position('bottom')

    plt.title(f'Macierz Pomyłek modelu (Dokładność: {accuracy}%)', pad=20, fontsize=15, fontweight='bold')
    plt.xlabel('Przewidziana klasa', fontsize=12)
    plt.ylabel('Rzeczywista klasa', fontsize=12)

    plt.tight_layout()
    plt.savefig(plik_wyjsciowy, dpi=300)
    print(f"Pomyślnie wygenerowano macierz pomyłek: '{plik_wyjsciowy}'.")
    plt.close()


rysuj_macierz_pomylek('test_log.txt')

generuj_wykres_z_logu('error_log.txt', 'wykres_bledu.png')