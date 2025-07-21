import pandas as pd

# Nome del file CSV da analizzare
filename = "gem5_results.csv"

df = pd.read_csv(filename)

# Conta i valori '1' nella colonna 'correct'
count_1 = (df['correct'] == 1).sum()
average_speedup = df['speedup'].mean()
count_optimized = ((df['correct'] == 1) & (df['speedup'] > 1)).mean() * 100
totale = len(df)
accuracy = (df['correct'] == 1).mean() * 100

# Prepara l'output
output = [
    f"File analizzato: {filename}",
    f"Speedup medio: {average_speedup}",
    f"Percentuale corretti: {accuracy:.2f}%",
    f"Percentuale ottimizzata: {count_optimized:.2f}%",
    f"Numero totale di righe: {totale}"
]

# Stampa su console
for line in output:
    print(line)

# Salva su file txt
with open("conteggio_correct_3_epochs.txt", "w") as f:
    for line in output:
        f.write(line + "\n")