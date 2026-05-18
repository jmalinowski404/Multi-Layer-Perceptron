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
    plt.xticks(range(0, max(epochs) + 1, 25), rotation=45)
    plt.ylabel('Błąd', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(nazwa_pliku_wyjsciowego, dpi=300)
    print(f"Pomyślnie wygenerowano wykres i zapisano jako: '{nazwa_pliku_wyjsciowego}'.")

    plt.close()

generuj_wykres_z_logu('log_lr02_m00.txt', 'wykres_bledu_lr02_m00.png')
generuj_wykres_z_logu('log_lr02_m09.txt', 'wykres_bledu_lr02_m09.png')
generuj_wykres_z_logu('log_lr06_m00.txt', 'wykres_bledu_lr06_m00.png')
generuj_wykres_z_logu('log_lr09_m00.txt', 'wykres_bledu_lr09_m00.png')
generuj_wykres_z_logu('log_lr09_m06.txt', 'wykres_bledu_lr09_m06.png')