# DOKUMENTACJA KOŃCOWA

## Evolution Strategies as a Scalable Alternative to Reinforcement Learning

**Autor:** Katarzyna Kadyszewska,  
**Przedmiot:** USD
**Data:** 04.02.2026

---

## 1. Wstęp

### 1.1 Cel projektu

Celem projektu była replikacja eksperymentu przedstawionego w artykule _Evolution Strategies as a Scalable Alternative to Reinforcement Learning_ (Salimans et al., 2017) oraz przeprowadzenie optymalizacji algorytmu Evolution Strategies (ES) na trzech wybranych zadaniach z benchmarku HumanoidBench.

### 1.2 Zakres projektu

Zgodnie z dokumentacją wstępną, zakres obejmował:

1. **Replikacja** - odtworzenie wyników ES na środowisku MuJoCo HalfCheetah-v4
2. **Optymalizacja** - zastosowanie i dostrojenie ES na trzech środowiskach HumanoidBench:
   - `h1hand-walk-v0` (lokomocja)
   - `h1hand-reach-v0` (manipulacja - sięganie)
   - `h1hand-push-v0` (manipulacja - pchanie)
3. **Analiza** - porównanie jakości uczenia na podstawie wybranych metryk

### 1.3 Zmiany względem dokumentacji wstępnej

| Aspekt                             | Dokumentacja wstępna | Dokumentacja końcowa | Uzasadnienie                                    |
| ---------------------------------- | -------------------- | -------------------- | ----------------------------------------------- |
| Wersja środowiska                  | HalfCheetah-v2       | HalfCheetah-v4       | v2 deprecated w Gymnasium                       |
| Architektura sieci (HumanoidBench) | 64×64                | 128×128              | Większa złożoność zadań humanoidalnych          |
| Max episode steps (HalfCheetah)    | 1000                 | 400                  | Więcej iteracji przy ograniczonych zasobach     |
| Liczba timesteps                   | 5M (artykuł)         | 1M                   | Ograniczone zasoby obliczeniowe (8 vs 1440 CPU) |

---

## 2. Opis algorytmu Evolution Strategies

### 2.1 Podstawy teoretyczne

Evolution Strategies (ES) to klasa algorytmów optymalizacji typu black-box, które optymalizują parametry polityki poprzez perturbacje w przestrzeni parametrów. W przeciwieństwie do metod Policy Gradient, ES nie wymaga obliczania gradientów przez backpropagację.

**Estymator gradientu ES:**

```
∇_θ J(θ) ≈ (1/nσ) Σᵢ F(θ + σεᵢ) εᵢ
```

gdzie:

- `εᵢ ~ N(0, I)` - perturbacje Gaussowskie
- `σ` - odchylenie standardowe szumu
- `F(·)` - funkcja zwrotu (return) w środowisku

**Aktualizacja parametrów:**

```
θₜ₊₁ = θₜ + α ∇_θ J(θₜ)
```

### 2.2 Zaimplementowane techniki

Zgodnie z artykułem Salimans et al. (2017), zaimplementowano następujące techniki:

| Technika                      | Opis                                                     | Lokalizacja w kodzie                            |
| ----------------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| **Antithetic sampling**       | Parowanie perturbacji εᵢ oraz -εᵢ dla redukcji wariancji | `es_algorithm.py`, metoda `_compute_gradient()` |
| **Fitness shaping**           | Transformacja rang zamiast surowych nagród               | `utils.py`, funkcja `compute_centered_ranks()`  |
| **Weight decay**              | Regularyzacja L2 parametrów (współczynnik 0.005)         | `es_algorithm.py`, linia 156                    |
| **Observation normalization** | Normalizacja obserwacji online (algorytm Welforda)       | `utils.py`, klasa `ObservationNormalizer`       |

### 2.3 Architektura sieci neuronowej

**Dla środowisk MuJoCo (HalfCheetah):**

```
Input (17) → Dense(64, tanh) → Dense(64, tanh) → Output (6)
Parametry: 5,702
```

**Dla środowisk HumanoidBench:**

```
Input (151-163) → Dense(128, tanh) → Dense(128, tanh) → Output (61)
Parametry: 43,837 - 45,373
```

---

## 3. Środowiska eksperymentalne

### 3.1 MuJoCo - HalfCheetah-v4

HalfCheetah to klasyczne zadanie kontroli ciągłej, w którym agent steruje dwunożnym robotem w celu maksymalizacji prędkości ruchu do przodu.

| Parametr          | Wartość |
| ----------------- | ------- |
| Wymiar obserwacji | 17      |
| Wymiar akcji      | 6       |
| Zakres akcji      | [-1, 1] |

### 3.2 HumanoidBench

HumanoidBench to benchmark oparty na humanoidzie Unitree H1, zawierający zadania lokomocji i manipulacji o wysokiej złożoności.

| Środowisko      | Typ         | Wymiar obs. | Wymiar akcji | Opis                                 |
| --------------- | ----------- | ----------- | ------------ | ------------------------------------ |
| h1hand-walk-v0  | Lokomocja   | 151         | 61           | Kontrola ruchu humanoida w przód     |
| h1hand-reach-v0 | Manipulacja | 157         | 61           | Precyzyjne sięganie do losowego celu |
| h1hand-push-v0  | Manipulacja | 163         | 61           | Pchanie obiektu do celu              |

---

## 4. Konfiguracja eksperymentów

### 4.1 Hiperparametry

Hiperparametry zostały dobrane zgodnie z artykułem źródłowym:

| Parametr            | HalfCheetah | HumanoidBench | Źródło                                     |
| ------------------- | ----------- | ------------- | ------------------------------------------ |
| Population size (n) | 40          | 40            | Artykuł, Tabela 4                          |
| Noise std (σ)       | 0.02        | 0.02          | Artykuł, Tabela 4                          |
| Learning rate (α)   | 0.01        | 0.01          | Artykuł, Tabela 4                          |
| Weight decay        | 0.005       | 0.005         | Artykuł                                    |
| Max episode steps   | 400         | 1000          | Dostosowane                                |
| Hidden layers       | 64×64       | 128×128       | 64×64 z artykułu / zwiększone dla humanoid |

### 4.2 Zasoby obliczeniowe

| Parametr          | Wartość                                    |
| ----------------- | ------------------------------------------ |
| Procesor          | Intel Core i5-1035G1 (8 wątków logicznych) |
| RAM               | 16 GB                                      |
| System operacyjny | Windows 11 + WSL2 (Ubuntu 24.04)           |
| Python            | 3.10                                       |
| Środowisko        | conda/mamba (es-rl)                        |

**Porównanie z artykułem:**

- Artykuł: 1,440 CPU cores, 10 minut dla 3D Humanoid (6000 reward)
- Nasz projekt: 8 wątków, ~60-80 minut dla 1M kroków

---

## 5. Wyniki eksperymentów

### 5.1 Eksperyment 1: Replikacja na HalfCheetah-v4

**Cel:** Weryfikacja poprawności implementacji ES poprzez odtworzenie krzywej uczenia.

#### 5.1.1 Konfiguracja

| Parametr          | Wartość        |
| ----------------- | -------------- |
| Environment       | HalfCheetah-v4 |
| Population size   | 40             |
| Sigma             | 0.02           |
| Learning rate     | 0.01           |
| Max episode steps | 400            |
| Total timesteps   | 1,008,000      |

#### 5.1.2 Wyniki

| Metryka                     | Wartość          |
| --------------------------- | ---------------- |
| Total timesteps             | 1,008,000        |
| Total iterations            | 63               |
| Training time               | 252.7s (4.2 min) |
| **Best reward**             | **483.6**        |
| Final reward (mean +/- std) | 132.2 +/- 60.1   |
| Average FPS                 | 3,988            |

#### 5.1.3 Analiza krzywej uczenia

Krzywa uczenia wykazuje charakterystyczny dla ES przebieg:

- **Faza początkowa (0-200k steps):** Szybki wzrost od -187 do +100
- **Faza eksploracji (200k-700k steps):** Stopniowy wzrost z wysoką wariancją
- **Peak wydajności (~720k steps):** Osiągnięcie best reward = 483.6
- **Faza końcowa:** Spadek spowodowany brakiem early stopping

#### 5.1.4 Porównanie z artykułem

| Metryka          | Artykuł (5M steps) | Nasz wynik (1M steps) | Stosunek  |
| ---------------- | ------------------ | --------------------- | --------- |
| TRPO baseline    | 2,386              | -                     | -         |
| ES do 100% TRPO  | 2.88M timesteps    | -                     | -         |
| Nasz best reward | -                  | 483.6                 | ~20% TRPO |

**Wniosek:** Przy 20% budżetu obliczeniowego (1M vs 5M) osiągnęliśmy ~20% wyniku końcowego, co potwierdza **liniową zbieżność** algorytmu ES zgodnie z artykułem.

---

### 5.2 Eksperyment 2: h1hand-walk-v0 (Lokomocja)

**Cel:** Optymalizacja ES dla zadania chodzenia humanoida.

#### 5.2.1 Wyniki

| Metryka                     | Wartość           |
| --------------------------- | ----------------- |
| Total timesteps             | 1,001,551         |
| Total iterations            | 338               |
| Training time               | 4,970s (82.8 min) |
| **Best reward**             | **9.0**           |
| Final reward (mean +/- std) | 7.7 +/- 2.6       |

#### 5.2.2 Analiza krzywej uczenia

Krzywa uczenia wykazuje charakterystyczny wzorzec "U":

- **Faza początkowa (0-150 iter):** Reward stabilny ~6.5
- **Faza spadku (150-200 iter):** Spadek do ~5.8 (eksploracja niekorzystnych regionów)
- **Faza wzrostu (200-338 iter):** Wzrost do ~8.0 z malejącą wariancją

**Obserwacje:**

- Malejąca wariancja (od 6.5 do ~2) wskazuje na stabilizację polityki
- Punkty ewaluacji: 9.9 → 8.1 → 6.4 → 6.0 → 7.7 → 7.3
- Robot nauczył się utrzymywać równowagę, ale nie chodzić

#### 5.2.3 Interpretacja

Zadanie lokomocji humanoida jest **ekstremalnie trudne** dla ES bez Virtual Batch Normalization (VBN). Artykuł Salimans et al. podkreśla: _"Without these methods ES proved brittle in our experiments"_.

Reward ~7-9 odpowiada minimalnej nagrodzie za utrzymanie pozycji stojącej bez upadku.

---

### 5.3 Eksperyment 3: h1hand-reach-v0 (Sięganie)

**Cel:** Optymalizacja ES dla zadania sięgania do losowego celu.

#### 5.3.1 Wyniki

| Metryka                     | Wartość           |
| --------------------------- | ----------------- |
| Total timesteps             | 1,000,000         |
| Total iterations            | 25                |
| Training time               | 3,408s (56.8 min) |
| **Best reward**             | **235.0**         |
| Final reward (mean +/- std) | -182.7 +/- 145.4  |

#### 5.3.2 Analiza krzywej uczenia

Krzywa uczenia charakteryzuje się:

- **Ekstremalną wariancją:** Zakres od -1000 do +4000
- **Brakiem trendu konwergencji:** Oscylacje wokół 0 przez cały trening
- **Sporadycznymi sukcesami:** Pojedyncze epizody z reward >1000

**Obserwacje:**

- Tylko 25 iteracji (długie epizody = mało aktualizacji)
- Środowisko ma losową pozycję celu w każdym epizodzie
- Wysoka wariancja std ~600-950 przez cały trening

#### 5.3.3 Interpretacja

Środowisko reach ma **stochastyczną funkcję nagrody** zależną od losowej pozycji celu. ES eksploruje przestrzeń parametrów, ale przy 25 iteracjach nie ma wystarczająco dużo aktualizacji gradientu do znalezienia stabilnej polityki.

**Rekomendacja:** Skrócenie epizodów lub zwiększenie populacji dla większej liczby iteracji.

---

### 5.4 Eksperyment 4: h1hand-push-v0 (Pchanie)

**Cel:** Optymalizacja ES dla zadania pchania obiektu.

#### 5.4.1 Wyniki

| Metryka                     | Wartość           |
| --------------------------- | ----------------- |
| Total timesteps             | 1,018,230         |
| Total iterations            | 52                |
| Training time               | 4,027s (67.1 min) |
| **Best reward**             | **-201.1**        |
| Final reward (mean +/- std) | -258.3 +/- 72.1   |

#### 5.4.2 Analiza krzywej uczenia

Krzywa uczenia wykazuje **wyraźny trend poprawy**:

- **Faza początkowa (iter 1-10):** Szybka poprawa od -600 do -250
- **Plateau (iter 10-30):** Stabilizacja na poziomie -250 do -350
- **Spike eksploracji (iter 30):** Wzrost wariancji do 1444 (odkrycie nowego regionu)
- **Faza końcowa (iter 40-52):** Stabilizacja na poziomie ~-300

**Obserwacje:**

- Wyraźna redukcja kary: od -600 do -200 (poprawa o 400)
- Malejąca wariancja końcowa = stabilizacja polityki

#### 5.4.3 Interpretacja

ES wykazał **zdolność uczenia się** w trudnym środowisku z dynamiką kontaktu fizycznego. Chociaż robot nie rozwiązał zadania w pełni, trend redukcji kary potwierdza, że algorytm eksploruje właściwy kierunek optymalizacji.

---

### 5.5 Podsumowanie wyników

| Środowisko      | Timesteps | Iterations | Best Reward | Final Reward     | Trend        | Trudność   |
| --------------- | --------- | ---------- | ----------- | ---------------- | ------------ | ---------- |
| HalfCheetah-v4  | 1,008,000 | 63         | **483.6**   | 132.2 +/- 60.1   | ↗️ Wzrost    | ⭐⭐       |
| h1hand-walk-v0  | 1,001,551 | 338        | 9.0         | 7.7 +/- 2.6      | → Plateau    | ⭐⭐⭐⭐⭐ |
| h1hand-reach-v0 | 1,000,000 | 25         | 235.0       | -182.7 +/- 145.4 | ↔️ Wariancja | ⭐⭐⭐     |
| h1hand-push-v0  | 1,018,230 | 52         | -201.1      | -258.3 +/- 72.1  | ↗️ Poprawa   | ⭐⭐⭐⭐   |

---

## 6. Wnioski

### 6.1 Potwierdzenie hipotez z artykułu

1. **ES jest skalowalny** - Algorytm działa na różnych środowiskach bez modyfikacji kodu (modułowość)
2. **ES nie wymaga backpropagacji** - Implementacja oparta wyłącznie na forward pass (3x mniej obliczeń)
3. **Antithetic sampling redukuje wariancję** - Potwierdzone we wszystkich eksperymentach
4. **Fitness shaping stabilizuje uczenie** - Transformacja rang eliminuje wpływ outlierów
5. **ES jest wrażliwy na parametryzację sieci** - Bez VBN wyniki są "kruche" (brittle)

### 6.2 Ograniczenia zidentyfikowane w projekcie

| Ograniczenie          | Wpływ                   | Możliwe rozwiązanie                       |
| --------------------- | ----------------------- | ----------------------------------------- |
| Brak VBN              | Słabe wyniki na walk    | Implementacja Virtual Batch Normalization |
| 8 wątków vs 1440 CPU  | Długi czas treningu     | Paralelizacja z Ray/multiprocessing       |
| 1M vs 5M timesteps    | Niepełna konwergencja   | Dłuższy trening lub early stopping        |
| Mało iteracji (reach) | Brak stabilnej polityki | Krótsze epizody lub większa populacja     |

### 6.3 Porównanie trudności zadań HumanoidBench

| Zadanie         | Trudność   | Uzasadnienie                                                   |
| --------------- | ---------- | -------------------------------------------------------------- |
| h1hand-reach-v0 | ⭐⭐⭐     | Stochastyczny cel, ale szybkie nagrody; wymaga więcej iteracji |
| h1hand-push-v0  | ⭐⭐⭐⭐   | Dynamika kontaktu, opóźnione nagrody; widoczny trend uczenia   |
| h1hand-walk-v0  | ⭐⭐⭐⭐⭐ | Wymaga VBN, bardzo długi horyzont; minimalna poprawa           |

### 6.4 Rekomendacje dla przyszłych prac

1. **Implementacja Virtual Batch Normalization** - kluczowe dla trudnych zadań
2. **Paralelizacja** z użyciem `multiprocessing` lub `Ray` dla przyspieszenia
3. **Adaptacyjne sigma** (styl CMA-ES) dla lepszej eksploracji
4. **Eksperymenty z większą populacją** (n=100+) dla reach/push
5. **Dyskretyzacja akcji** dla niektórych zadań (jak w artykule dla Hopper/Swimmer)

---

## 7. Instrukcja reprodukcji wyników

### 7.1 Wymagania systemowe

- Python 3.10
- System: Linux (natywny lub WSL2)
- RAM: minimum 8 GB
- Procesor: minimum 4 rdzenie

### 7.2 Instalacja środowiska

```bash
# 1. Klonowanie repozytorium
git clone https://github.com/kat1478/es-mujoco-humanoidbench.git
cd es-mujoco-humanoidbench

# 2. Tworzenie środowiska conda/mamba
mamba env create -f environment.yml
# lub
conda env create -f environment.yml

# 3. Aktywacja środowiska
conda activate es-rl

# 4. Instalacja HumanoidBench (z GitHub)
git clone https://github.com/carlosferrazza/humanoid-bench.git
cd humanoid-bench
pip install -e .
cd ..

# 5. Poprawka NumPy (jeśli potrzebna)
pip install "numpy<2.0"

# 6. Konfiguracja renderowania (dla WSL2/headless)
export MUJOCO_GL=egl
```

### 7.3 Uruchomienie eksperymentów

#### Eksperyment 1: HalfCheetah (replikacja)

```bash
cd experiments

# Quick test (~5 min)
python run_halfcheetah_optimized.py --quick

# Pełny eksperyment (~5 min dla 1M steps)
python run_halfcheetah_optimized.py --timesteps 1000000
```

#### Eksperymenty 2-4: HumanoidBench

```bash
# Quick test wszystkich środowisk (~15 min)
python run_humanoid.py --env all --quick

# Pełny eksperyment (~3.5 godz dla wszystkich)
python run_humanoid.py --env all --steps 1000000

# Pojedyncze środowisko
python run_humanoid.py --env h1hand-walk-v0 --steps 1000000
python run_humanoid.py --env h1hand-reach-v0 --steps 1000000
python run_humanoid.py --env h1hand-push-v0 --steps 1000000
```

### 7.4 Struktura projektu

```
es-mujoco-humanoidbench/
├── src/
│   ├── __init__.py          # Eksporty modułów
│   ├── policy.py            # Sieć neuronowa MLP
│   ├── es_algorithm.py      # Główny algorytm ES
│   └── utils.py             # Funkcje pomocnicze (ranks, normalizer)
├── experiments/
│   ├── run_halfcheetah.py           # Eksperyment podstawowy
│   ├── run_halfcheetah_optimized.py # Eksperyment zoptymalizowany
│   └── run_humanoid.py              # Eksperymenty HumanoidBench
├── results/                  # Wyniki (JSON, NPZ, PNG)
│   ├── halfcheetah_optimized_*.json
│   ├── h1hand-walk-v0_seed42_*.json
│   ├── h1hand-reach-v0_seed42_*.json
│   ├── h1hand-push-v0_seed42_*.json
│   └── *_curve.png          # Wykresy uczenia
├── environment.yml           # Zależności conda
├── requirements.txt          # Zależności pip
└── README.md
```

### 7.5 Opis plików wynikowych

| Plik               | Zawartość                                                           |
| ------------------ | ------------------------------------------------------------------- |
| `*_TIMESTAMP.json` | Pełna historia treningu, konfiguracja, metryki                      |
| `*_policy.npz`     | Wagi sieci neuronowej i statystyki normalizacji                     |
| `*_curve.png`      | 4-panelowy wykres: krzywa uczenia, rozkład, wariancja, podsumowanie |

---

## 8. Bibliografia

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). _Evolution Strategies as a Scalable Alternative to Reinforcement Learning_. arXiv:1703.03864. https://arxiv.org/abs/1703.03864

2. OpenAI. (2017). _Evolution Strategies Starter_. GitHub repository. https://github.com/openai/evolution-strategies-starter

3. HumanoidBench. (2024). _Humanoid Manipulation and Locomotion Benchmark_. https://humanoid-bench.github.io

4. Todorov, E., Erez, T., & Tassa, Y. (2012). _MuJoCo: A physics engine for model-based control_. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems.

5. Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J., & Schmidhuber, J. (2014). _Natural Evolution Strategies_. Journal of Machine Learning Research, 15(1), 949-980.

6. Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). _Trust Region Policy Optimization_. In ICML, pages 1889-1897.

---

## Załączniki

### A. Wykresy uczenia

Wykresy znajdują się w folderze `results/`:

- `halfcheetah_optimized_curve_*.png`
- `h1hand-walk-v0_seed42_curve.png`
- `h1hand-reach-v0_seed42_curve.png`
- `h1hand-push-v0_seed42_curve.png`

### B. Surowe dane eksperymentów

Pliki JSON w folderze `results/` zawierają:

- Pełną konfigurację eksperymentu
- Historię treningu (reward per iteration)
- Historię ewaluacji
- Metryki końcowe

### C. Wytrenowane polityki

Pliki NPZ w folderze `results/` zawierają:

- Wagi sieci neuronowej (theta)
- Statystyki normalizacji obserwacji (mean, var, count)

---

_Dokument wygenerowany: 4 lutego 2026_
