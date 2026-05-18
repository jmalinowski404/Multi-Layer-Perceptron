import matplotlib.pyplot as plt
import re

etykiety = ['Wzorzec 1 [1,0,0,0]', 'Wzorzec 2 [0,1,0,0]', 'Wzorzec 3 [0,0,1,0]', 'Wzorzec 4 [0,0,0,1]']

no_bias_x = [0.111, 0.997, 0.011, 0.111]
no_bias_y = [0.111, 0.012, 0.997, 0.111]

bias_x = [0.857, 0.178, 0.988, 0.013]
bias_y = [0.987, 0.009, 0.134, 0.720]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(no_bias_x, no_bias_y, color='red', s=120, edgecolors='black', zorder=5)

for i, txt in enumerate(etykiety):
    if i == 0:
        ax1.annotate(txt, (no_bias_x[i], no_bias_y[i]), xytext=(10, 10), textcoords='offset points')
    elif i == 3:
        ax1.annotate(txt, (no_bias_x[i], no_bias_y[i]), xytext=(10, -15), textcoords='offset points')
    else:
        ax1.annotate(txt, (no_bias_x[i], no_bias_y[i]), xytext=(10, 5), textcoords='offset points')

ax1.set_title('Bez biasu', fontsize=14, fontweight='bold')
ax1.set_xlim(-0.1, 1.2)
ax1.set_ylim(-0.1, 1.2)
ax1.set_xlabel('Wyjście Neuronu Ukrytego 1', fontsize=12)
ax1.set_ylabel('Wyjście Neuronu Ukrytego 2', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.scatter(bias_x, bias_y, color='green', s=120, edgecolors='black', zorder=5)

for i, txt in enumerate(etykiety):
    ax2.annotate(txt, (bias_x[i], bias_y[i]), xytext=(10, 5), textcoords='offset points')

ax2.set_title('Z biasem', fontsize=14, fontweight='bold')
ax2.set_xlim(-0.1, 1.2)
ax2.set_ylim(-0.1, 1.2)
ax2.set_xlabel('Wyjście Neuronu Ukrytego 1', fontsize=12)
ax2.set_ylabel('Wyjście Neuronu Ukrytego 2', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.suptitle('Warstwy ukryte autoenkodera', fontsize=16, fontweight='bold')
plt.tight_layout()

plt.savefig('przestrzen_ukryta_autoenkodera.png', dpi=300)
plt.close()

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

generuj_wykres_z_logu('error_log_bias.txt', 'wykres_bledu_bias.png')
generuj_wykres_z_logu('error_log_no_bias.txt', 'wykres_bledu_no_bias.png')