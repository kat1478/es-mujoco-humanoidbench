# Evolution Strategies as a Scalable Alternative to Reinforcement Learning

Implementacja algorytmu Evolution Strategies (ES) na podstawie artykułu:

> Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017).
> _Evolution Strategies as a Scalable Alternative to Reinforcement Learning_.
> arXiv:1703.03864

## Autorzy projektu

- Katarzyna Kadyszewska

## Cel projektu

1. **Replikacja** wyników ES na środowisku MuJoCo HalfCheetah-v4
2. **Optymalizacja** algorytmu ES na trzech środowiskach HumanoidBench:
   - h1hand-walk-v0
   - h1hand-reach-v0
   - h1hand-push-v0

## Instalacja

### Wymagania

- Python 3.10
- Mamba lub Conda

### Tworzenie środowiska

```bash
# Używając mamby (rekomendowane)
mamba env create -f environment.yml

# Lub używając condy
conda env create -f environment.yml

# Aktywacja środowiska
mamba activate es-rl
# lub
conda activate es-rl
```

## Struktura projektu

```
es-mujoco-humanoidbench/
├── src/
│   ├── policy.py           # Sieć neuronowa (MLP)
│   ├── es_algorithm.py     # Główny algorytm ES
│   └── utils.py            # Funkcje pomocnicze
├── experiments/
│   ├── run_halfcheetah.py  # Eksperyment replikacji
│   └── run_humanoid.py     # Eksperymenty HumanoidBench
├── results/                # Wyniki eksperymentów
├── configs/                # Konfiguracje
├── environment.yml         # Zależności conda/mamba
└── README.md
```

## Uruchomienie

### Szybki test

```bash
cd experiments
python run_halfcheetah.py --quick
```

### Pełny eksperyment na HalfCheetah

```bash
python run_halfcheetah.py --timesteps 500000
```

### Parametry

- `--timesteps`: Liczba kroków treningowych (domyślnie: 500000)
- `--population`: Rozmiar populacji (domyślnie: 40)
- `--sigma`: Odchylenie standardowe szumu (domyślnie: 0.02)
- `--lr`: Learning rate (domyślnie: 0.01)
- `--seed`: Ziarno losowości (domyślnie: 42)

## Algorytm ES

Evolution Strategies optymalizuje parametry polityki poprzez:

1. **Perturbacje parametrów**: Dodanie szumu gaussowskiego do wag sieci
2. **Ewaluacja**: Ocena każdej wersji polityki w środowisku
3. **Estymacja gradientu**: Obliczenie kierunku poprawy na podstawie nagród
4. **Aktualizacja**: Zmiana parametrów w kierunku gradientu

### Kluczowe techniki:

- **Antithetic sampling**: Parowanie perturbacji ε i -ε dla redukcji wariancji
- **Fitness shaping**: Rangowanie nagród dla stabilności
- **Normalizacja obserwacji**: Skalowanie wejść dla lepszej generalizacji

## Wyniki

_(Będą uzupełnione po przeprowadzeniu eksperymentów)_

## Bibliografia

1. Salimans, T., et al. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864
2. OpenAI. (2017). Evolution Strategies Starter. GitHub repository.
3. HumanoidBench. (2024). Humanoid Manipulation and Locomotion Benchmark.
