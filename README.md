# Využitie hlbokého učenia pre spracovanie a analýzu ionosférických dát

*Diplomová práca*

### Bc. Silvia Kostárová

*Technická univerzita v Košiciach\
Fakulta elektrotechniky a informatiky\
Študijný program: Hospodárska informatika\
Školiace pracovisko: Katedra kybernetiky a umelej inteligencie*

## 1. Funkcia programu
Funkcie programu v podobe skriptov zahŕňajú prípravu dát, trénovanie modelov, predikcia z testovacích dát, spracovanie výstupov z modelov, vyhodnotenie modelov podľa metrík, kvalitatívne vyhodnotenie.

### 1. 1. Generovanie časových okien
Ide o prípravu dát na trénovanie a testovanie modelov. My sme použili na testovanie dáta za rok 2019 a na trénovanie roky od 2013 do 2021 okrem testovacieho roku (toto nastavenie je možné meniť).

Na generovanie časových okien pre trénovaciu a testovaciu množinu slúži skript [`data/data_preparation_v2_2024.ipynb`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/data/data_preparation_v2_2024.ipynb). Dáta sú generované z jednotlivých binov, ktoré sa nachádzajú [tu](https://mega.nz/folder/5r5iQIaC#4myXsED61CcgIvdIiYZrhA). Tieto dáta zahŕňajú údaje parametrov $\sigma_\phi$, *ASY/H*, *Bz GSE*, *PC(N)*. Hodinové údaje o *Ap index* sa nachádzajú v súbore [`datasets/omni4col_INTER_NORMALIZED_hour_2013-2021.parquet.gzip`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/datasets/omni4col_INTER_NORMALIZED_hour_2013-2021.parquet.gzip).

Proces generovania dát je založený na prechádzaní časovým radom a hľadaní okien stanovenej veľkosti. Po vygenerovaní okien sa vybalansuje trénovacia množina. Vytvorené množiny sa ukladajú do štruktúry <code>*{model_name}*/shift-*{shift_size}*-windows-*{windows_size}*/</code> ako je uvedené [tu](https://mega.nz/folder/NWdAxaha#VyY9R_i9CcMmEBdqWG77cw). Na tomto odkaze sa zároveň nachádzajú všetky vytvorené datasety, ktoré boli použité v tejto práci na trénovanie a testovanie. Obsah jedného priečinku k jednej kombinácii model-shift-windows je nasledovný:
- `X_train_hour.npy` - trénovacie dáta $X$ obsahujúce hodinové údaje,
- `X_train_min.npy` - trénovacie dáta $X$ obsahujúce minútové údaje,
- `X_test_hour.npy` - testovacie dáta $X$ obsahujúce hodinové údaje,
- `X_test_min.npy` - testovacie dáta $X$ obsahujúce minútové  údaje,
- `y_test.npy` - cieľové testovacie hodnoty $y$,
- `y_train.npy` - cieľové trénovacie hodnoty $y$,
- `test_index_bin.npy` - informácia o bine pre každý testovací záznam,
- `test_index_timestamp.npy` - informácia o čase pre každý testovací záznam.

### 1. 2. Trénovanie modelov
Pre trénovanie modelov slúžia nasledujúce skripty nachádzajúce sa v priečinku [model](https://github.com/skostarova/Diplomovy_projekt_Kostarova/tree/main/model):
- [`autoreg.py`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/autoreg.py) - trénovanie autoregresných modelov,
- [`train_multiNN_v1a.py`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/train_multiNN_v1a.py) - trénovanie základného multivariačného modelu s pridaním 1 minútového parametra,
- [`train_multiNN_v1b.py`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/train_multiNN_v1b.py) - trénovanie základného multivariačného modelu s pridaním 1 hodinového parametra,
- [`train_multiNN_v2.py`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/train_multiNN_v2.py) - trénovanie rozšíreného multivariačného modelu s pridanými všetkými parametrami na vstupe.

Pomocou týchto skriptov sa ukladajú modely z každej epochy ako <code>newModel/*model_name*/shift-*shift_size*-windows-*windows_size*/epoch-*{epoch}*-mae-*{val_mae}*.hdf5</code>. Najlepšie modely vytvorené počas nášho trénovania, ktoré patria medzi finálne výstupy tejto práce, sú uložené v priečinku [model](https://github.com/skostarova/Diplomovy_projekt_Kostarova/tree/main/model) s upravenými názvami, ktoré pozostávajú z názvu modelu (resp. parametra, ktorý bol použitý) a hodnotách `shift` a `windows`:
- [`autoreg-shift-15-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/autoreg-shift-15-windows-45.hdf5)
- [`autoreg-shift-30-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/autoreg-shift-30-windows-45.hdf5)
- [`autoreg-shift-45-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/autoreg-shift-45-windows-45.hdf5)
- [`autoreg-shift-60-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/autoreg-shift-60-windows-45.hdf5)
- [`ap_index-shift-15-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/ap_index-shift-15-windows-45.hdf5)
- [`AsyH-shift-15-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/AsyH-shift-15-windows-45.hdf5)
- [`BzGSE-shift-15-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/BzGSE-shift-15-windows-45.hdf5)
- [`PC-shift-15-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/PC-shift-15-windows-45.hdf5)
- [`all_parameters-shift-15-windows-45.hdf5`](https://github.com/skostarova/Diplomovy_projekt_Kostarova/blob/main/model/all_parameters-shift-15-windows-45.hdf5)

### 1. 3. Predikcie
Na vytvorenie predikovaných hodnôt z modelov uvedených v priečinku [model](https://github.com/skostarova/Diplomovy_projekt_Kostarova/tree/main/model) slúži skript

### 1. 4. Post-processing

### 1. 5. Vyhodnotenie

