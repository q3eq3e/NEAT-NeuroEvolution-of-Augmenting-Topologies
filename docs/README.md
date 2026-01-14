# Dokumentacja Architektoniczna - NEAT (NeuroEvolution of Augmenting Topologies)

## Spis treści
1. [Przegląd projektu](#przegląd-projektu)
2. [Struktura folderów](#struktura-folderów)
3. [Architektura systemu](#architektura-systemu)
4. [Moduły i klasy](#moduły-i-klasy)
5. [Przepływ danych](#przepływ-danych)
6. [Wywołanie głównego programu](#wywołanie-głównego-programu)
7. [Diagram zależności](#diagram-zależności)
8. [Konfiguracja i parametry](#konfiguracja-i-parametry)

---

## Przegląd projektu

Projekt implementuje algorytm **NEAT (NeuroEvolution of Augmenting Topologies)** - zaawansowaną metodę ewolucyjną do tworzenia i trenowania sztucznych sieci neuronowych. NEAT łączy w sobie ewolucję topologii sieci (struktury) oraz wag połączeń, umożliwiając automatyczne odkrywanie optymalnych architektur sieci neuronowych dla danego problemu.

### Główne cechy implementacji:
- **Ewolucja topologii**: Dynamiczne dodawanie węzłów i połączeń
- **Specjacja**: Grupowanie podobnych genomów w gatunki
- **Innowacje historyczne**: Śledzenie mutacji strukturalnych
- **Integracja z Gymnasium**: Wsparcie dla środowisk OpenAI Gym
- **Logging i monitoring**: Śledzenie postępów treningu

---

## Struktura folderów

```
golem/
├── src/                          # Kod źródłowy projektu
│   ├── modeling/                 # Moduły związane z algorytmem NEAT
│   │   ├── __init__.py
│   │   ├── neat.py              # Główna klasa algorytmu NEAT
│   │   ├── genome.py            # Reprezentacja genomu
│   │   ├── nn.py                # Klasa sieci neuronowej
│   │   ├── node.py              # Węzły i połączenia sieci
│   │   └── activation.py        # Funkcje aktywacji
│   ├── environment.py           # Wrapper dla środowisk Gymnasium
│   ├── logger.py                # Klasy do logowania metryk
│   ├── config.py                # Konfiguracja projektu
│   └── __init__.py
├── example/                      # Przykłady użycia
│   └── main.py                  # Główny skrypt treningowy
├── models/                       # Zapisane modele (.pkl)
├── docs/                         # Dokumentacja
├── reports/                      # Raporty i wykresy
└── requirements.txt             # Zależności projektu
```

---

## Architektura systemu

System NEAT składa się z kilku warstw abstrakcji:

### Warstwa 1: Reprezentacja sieci neuronowej
- **Node**: Pojedynczy neuron w sieci
- **Connection**: Połączenie między neuronami
- **NN**: Kompletna sieć neuronowa z możliwością obliczania wyjścia

### Warstwa 2: Reprezentacja genetyczna
- **Genome**: Genotyp reprezentujący sieć neuronową
- Zawiera listę genów (połączeń) i odniesienie do fenotypu (NN)

### Warstwa 3: Algorytm ewolucyjny
- **NEAT**: Główny silnik ewolucyjny
- Zarządza populacją, mutacjami, krzyżowaniem i specjacją

### Warstwa 4: Integracja ze środowiskiem
- **Environment**: Wrapper dla środowisk Gymnasium
- Obsługuje różne typy przestrzeni akcji (dyskretne/ciągłe)

### Warstwa 5: Monitoring i logging
- **Logger**: Interfejs bazowy
- **FitnessLogger**: Logowanie fitness przez generacje
- **SpeciesLogger**: Logowanie liczby gatunków

---

## Moduły i klasy

### 1. `src/modeling/node.py`

#### Klasa `NodeTypes` (Enum)
Definiuje typy węzłów w sieci:
- `INPUT = 1`: Węzły wejściowe
- `HIDDEN = 2`: Węzły ukryte
- `OUTPUT = 3`: Węzły wyjściowe

#### Klasa `Node`
Reprezentuje pojedynczy neuron w sieci neuronowej.

**Atrybuty:**
- `index`: Unikalny identyfikator węzła
- `type`: Typ węzła (NodeTypes)
- `bias`: Wartość biasu
- `act`: Funkcja aktywacji
- `layer`: Numer warstwy (0 dla INPUT, 1 dla OUTPUT)
- `connections`: Lista połączeń wejściowych
- `out`: Ostatnia obliczona wartość wyjściowa

**Kluczowe metody:**
- `add_input(input_node, weight, innovation_number)`: Dodaje połączenie wejściowe
- `calculate_output()`: Oblicza wyjście węzła na podstawie wejść
- `get_active_connections()`: Zwraca aktywne połączenia

#### Klasa `Connection`
Reprezentuje połączenie między dwoma węzłami.

**Atrybuty:**
- `from_node`: Węzeł źródłowy
- `to_node`: Węzeł docelowy
- `weight`: Waga połączenia
- `innovation_number`: Unikalny numer innowacji
- `enabled`: Czy połączenie jest aktywne

**Kluczowe metody:**
- `get_weight()` / `set_weight(weight)`: Getter/setter wagi
- `disable()` / `enable()`: Wyłączanie/włączanie połączenia
- `get_source_node()` / `get_target_node()`: Gettery węzłów

---

### 2. `src/modeling/nn.py`

#### Klasa `NN`
Reprezentuje sieć neuronową jako fenotyp genomu.

**Atrybuty:**
- `input_size`: Liczba wejść
- `output_size`: Liczba wyjść
- `act`: Domyślna funkcja aktywacji
- `connections`: Lista wszystkich połączeń
- `nodes`: Lista wszystkich węzłów

**Kluczowe metody:**

##### `__init__(input_size, output_size, act=sigmoid)`
Tworzy minimalną sieć z pełnym połączeniem między wejściami a wyjściami.

##### `create_from_parent(parent, infos)` (statyczna)
Tworzy nową sieć na podstawie rodzica z zaktualizowanymi wagami.

##### `add_connection(from_node, to_node, innovation_number, weight)`
Dodaje nowe połączenie między węzłami.

##### `add_node(connection, innovation, act, bias, out)`
Dodaje nowy węzeł, dzieląc istniejące połączenie:
1. Wyłącza stare połączenie
2. Tworzy nowy węzeł ukryty
3. Dodaje dwa nowe połączenia (wejście→nowy węzeł, nowy węzeł→wyjście)
4. Dostosowuje warstwy sieci

##### `_adjust_layers(connection_from, connection_to)`
Zarządza warstwami sieci przy dodawaniu nowego węzła. Obsługuje różne przypadki:
- Różnica warstw > 1: Wstawia węzeł między warstwy
- Różnica warstw = 1: Tworzy nową warstwę
- Różnica warstw ≤ 0: Specjalne przypadki (pętle, połączenia wsteczne)

##### `calculate_output(inputs)`
Oblicza wyjście sieci:
1. Ustawia wartości węzłów wejściowych
2. Przetwarza węzły warstwa po warstwie (forward pass)
3. Zwraca wartości węzłów wyjściowych

##### `__str__()`
Generuje wizualizację ASCII sieci neuronowej z połączeniami.

---

### 3. `src/modeling/genome.py`

#### Klasa `Genome`
Reprezentuje genotyp - genetyczną reprezentację sieci neuronowej.

**Atrybuty:**
- `fitness`: Wartość fitness genomu
- `nn`: Odniesienie do obiektu NN (fenotyp)
- `_genes`: Lista połączeń (geny)

**Konstruktor:** `Genome(nn)`
Tworzy genom wskazujący na istniejący już obiekt NN (bez kopiowania):

**Kluczowe metody:**

##### `create_from_parent(parent, info_genes)` (statyczna)
Tworzy genom z rodzica i informacji o genach do zmiany.

##### `get_genes()` / `get_active_genes()`
Zwraca wszystkie/aktywne geny (połączenia).

##### `_create_nn(parent, info)`
Tworzy nową sieć neuronową na podstawie rodzica i informacji o genach do zmiany.

##### `predict(input)`
Wykonuje predykcję - deleguje do `nn.calculate_output()`.

---

### 4. `src/modeling/neat.py`

#### Klasa `NEAT`
Główny silnik algorytmu ewolucyjnego.

**Atrybuty:**
- `input_size` / `output_size`: Rozmiary wejścia/wyjścia
- `population`: Lista genomów w populacji
- `species`: Lista gatunków (listy genomów)
- `innovation_number`: Globalny licznik innowacji

**Kluczowe metody:**

##### `initialize_population(size)`
Tworzy początkową populację z minimalnymi sieciami (fully connected).

##### `get_new_innovation_number()`
Zwraca i inkrementuje globalny licznik innowacji.

##### Operatory genetyczne:

###### `crossover(parent1, parent2)`
Krzyżowanie dwóch genomów:
1. Identyfikuje geny: matching, disjoint, excess
2. Dziedziczy matching genes losowo od rodziców
3. Dziedziczy disjoint/excess genes od lepszego rodzica
4. Tworzy nowy genom potomny

###### `mutate_weights(genome, weight_mutation_rate, mutation_range)`
Mutacja wag połączeń:
- Z prawdopodobieństwem `weight_mutation_rate` mutuje każdą wagę
- Dodaje losową wartość z zakresu `[-mutation_range, mutation_range]`

###### `mutate_add_node(genome, innovation_number)`
Mutacja strukturalna - dodanie węzła:
1. Wybiera losowe aktywne połączenie
2. Dzieli je, dodając nowy węzeł ukryty
3. Przypisuje numer innowacji

###### `mutate_add_connection(genome, innovation_number, mutation_range)`
Mutacja strukturalna - dodanie połączenia:
1. Wybiera dwa losowe węzły
2. Sprawdza czy połączenie już istnieje
3. Dodaje nowe połączenie z losową wagą

###### `_mutate(genome, ...)`
Orkiestruje wszystkie typy mutacji dla pojedynczego genomu.

##### Specjacja:

###### `delta(genome1, genome2, c1, c2, c3)`
Oblicza dystans genetyczny między genomami:
```
δ = (c1 * E / N) + (c2 * D / N) + c3 * W̄
```
gdzie:
- E: liczba genów excess
- D: liczba genów disjoint
- W̄: średnia różnica wag matching genes
- N: liczba genów w większym genomie
- c1, c2, c3: współczynniki wagowe

###### `speciate(c1, c2, c3, compatibility_threshold)`
Grupuje populację w gatunki:
1. Dla każdego genomu oblicza dystans do reprezentantów gatunków
2. Przypisuje do pierwszego gatunku z δ < threshold
3. Jeśli nie pasuje do żadnego, tworzy nowy gatunek

##### Reprodukcja:

###### `determine_offspring()`
Oblicza liczbę potomków dla każdego gatunku proporcjonalnie do średniego fitness.

###### `reproduce(best_individuals_copied)`
Tworzy nową generację:
1. Kopiuje najlepsze osobniki (elityzm)
2. Dla każdego gatunku:
   - Sortuje według fitness
   - Wykonuje krzyżowanie najlepszych osobników
   - Mutuje potomków

##### Trening:

###### `train(evaluate, ...)`
Główna pętla treningowa:
```python
for generation in range(num_generations):
    1. Ewaluacja fitness dla każdego genomu
    2. Specjacja populacji
    3. Reprodukcja (krzyżowanie + mutacja)
    4. Logging i monitoring
```

**Parametry treningu:**
- `evaluate`: Funkcja ewaluacji fitness
- `weight_mutation_rate`: Prawdopodobieństwo mutacji wag (0.3)
- `mutation_range`: Zakres mutacji wag (0.5)
- `add_node_rate`: Prawdopodobieństwo dodania węzła (0.01)
- `add_connection_rate`: Prawdopodobieństwo dodania połączenia (0.01)
- `compatibility_threshold`: Próg dystansu dla specjacji (3)
- `c1, c2, c3`: Współczynniki dystansu genetycznego (0.5, 2, 2)
- `best_individuals_copied`: Procent elity (0.1)
- `num_generations`: Liczba generacji (100)
- `population_size`: Rozmiar populacji (2000)
- `act`: Funkcja aktywacji (sigmoid)
- `callbacks`: Lista loggerów

##### `get_best()`
Zwraca genom z najwyższym fitness.

---

### 5. `src/modeling/activation.py`

Moduł zawiera funkcje aktywacji dla neuronów:

#### `identity(x)`
Funkcja tożsamościowa: f(x) = x
- Używana dla węzłów wejściowych

#### `sigmoid(x)`
Funkcja sigmoidalna: f(x) = 1 / (1 + e^(-x))
- Zakres: (0, 1)
- Domyślna funkcja aktywacji

#### `tanh(x)`
Tangens hiperboliczny: f(x) = tanh(x)
- Zakres: (-1, 1)
- Alternatywa dla sigmoid

---

### 6. `src/environment.py`

#### Klasa `Environment`
Wrapper dla środowisk Gymnasium zapewniający jednolity interfejs.

**Atrybuty:**
- `env`: Obiekt środowiska Gymnasium
- `seed`: Ziarno losowości dla reprodukowalności

**Enum `ActionType`:**
- `DISCRETE`: Przestrzeń akcji dyskretna (np. Atari)
- `CONTINUOUS`: Przestrzeń akcji ciągła (np. MountainCar)
- `OTHER`: Nieobsługiwany typ

**Kluczowe metody:**

##### `get_params_num()`
Zwraca rozmiary przestrzeni obserwacji i akcji:
- Dla dyskretnych: (obs_size, num_actions)
- Dla ciągłych: (obs_size, action_dim)

##### `run_env(model, store_gif=False)`
Wykonuje jeden epizod w środowisku:
1. Resetuje środowisko
2. W pętli:
   - Pobiera akcję z modelu
   - Konwertuje akcję (argmax dla dyskretnych, skalowanie dla ciągłych)
   - Wykonuje krok w środowisku
   - Akumuluje nagrodę
3. Opcjonalnie zapisuje klatki do GIF
4. Zwraca całkowitą nagrodę (i klatki)

#### Funkcja `store_episode_as_gif(frames, path, filename)`
Zapisuje sekwencję klatek jako animowany GIF używając matplotlib.

---

### 7. `src/logger.py`

#### Klasa `Logger` (bazowa)
Interfejs dla loggerów z metodą `log(generation, stats)`.

#### Klasa `FitnessLogger`
Loguje wartości fitness przez generacje.

**Metody:**
- `log(generation, stats)`: Zapisuje `stats["fitness"]`
- `get_logs()`: Zwraca historię fitness
- `save_chart_data(filepath)`: Generuje wykres fitness vs generacje

#### Klasa `SpeciesLogger`
Loguje informacje o gatunkach.

**Metody:**
- `log(generation, stats)`: Zapisuje `stats["species"]`
- `get_logs()`: Zwraca historię gatunków
- `save_to_csv(filepath)`: Eksportuje dane do CSV (separator: `;`)

---

### 8. `src/config.py`

Moduł konfiguracyjny wykorzystujący zmienne środowiskowe.

**Zmienne:**
- `ENV_NAME`: Nazwa środowiska Gymnasium (domyślnie: "MountainCarContinuous-v0")

Konfiguracja jest ładowana z pliku `.env` używając `python-dotenv`.

---

## Przepływ danych

### Schemat przepływu podczas treningu:

```
1. Inicjalizacja
   ├─> NEAT.__init__(input_size, output_size)
   ├─> NEAT.initialize_population(size)
   │   └─> Dla każdego genomu:
   │       ├─> NN.__init__() - tworzy minimalną sieć
   │       └─> Genome(genes, nn) - opakowuje sieć

2. Pętla treningowa (dla każdej generacji)
   │
   ├─> Ewaluacja fitness
   │   ├─> Dla każdego genomu w populacji:
   │   │   ├─> Environment.run_env(genome)
   │   │   │   ├─> env.reset()
   │   │   │   └─> Pętla epizodu:
   │   │   │       ├─> genome.predict(observation)
   │   │   │       │   └─> NN.calculate_output(inputs)
   │   │   │       │       └─> Forward pass przez warstwy
   │   │   │       ├─> env.step(action)
   │   │   │       └─> Akumulacja nagrody
   │   │   └─> genome.fitness = total_reward
   │
   ├─> Specjacja
   │   ├─> NEAT.speciate(c1, c2, c3, threshold)
   │   │   └─> Dla każdego genomu:
   │   │       ├─> Oblicz delta() do reprezentantów
   │   │       └─> Przypisz do gatunku lub utwórz nowy
   │
   ├─> Reprodukcja
   │   ├─> NEAT.determine_offspring()
   │   │   └─> Oblicz liczbę potomków na gatunek
   │   │
   │   └─> NEAT.reproduce(best_individuals_copied)
   │       ├─> Elityzm: kopiuj najlepsze osobniki
   │       │
   │       └─> Dla każdego gatunku:
   │           ├─> Krzyżowanie:
   │           │   ├─> Wybierz dwóch rodziców
   │           │   └─> NEAT.crossover(parent1, parent2)
   │           │       ├─> Identyfikuj matching/disjoint/excess genes
   │           │       ├─> Dziedzicz geny
   │           │       └─> Utwórz nowy Genome
   │           │
   │           └─> Mutacja:
   │               ├─> NEAT._mutate(genome, ...)
   │               │   ├─> mutate_weights()
   │               │   ├─> mutate_add_node()
   │               │   └─> mutate_add_connection()
   │               └─> Aktualizuj NN strukturę
   │
   └─> Logging
       ├─> FitnessLogger.log(gen, {"fitness": best_fitness})
       └─> SpeciesLogger.log(gen, {"species": species_sizes})

3. Po treningu
   ├─> NEAT.get_best() - zwróć najlepszy genom
   ├─> Zapisz model (pickle)
   └─> Ewaluacja z wizualizacją (store_gif=True)
```

### Przepływ danych w sieci neuronowej:

```
Input (observation)
    ↓
[Node INPUT 0] [Node INPUT 1] ... [Node INPUT n]
    ↓               ↓                   ↓
    └───────────────┴───────────────────┘
                    ↓
        [Connections z wagami]
                    ↓
    ┌───────────────┴───────────────────┐
    ↓               ↓                   ↓
[Node HIDDEN]  [Node HIDDEN]  ...  (opcjonalne)
    ↓               ↓
    └───────────────┴───────────────────┐
                    ↓
        [Connections z wagami]
                    ↓
    ┌───────────────┴───────────────────┐
    ↓               ↓                   ↓
[Node OUTPUT 0] [Node OUTPUT 1] ... [Node OUTPUT m]
    ↓
Output (action)
```

**Obliczanie wyjścia węzła:**
```
output = activation(Σ(weight_i * input_i) + bias)
```

---

## Wywołanie głównego programu

### Struktura `example/main.py`

Główny skrypt treningowy obsługuje dwa tryby działania:

#### 1. Tryb treningowy (bez argumentów)

```bash
python3 -m example.main
```

**Przepływ wykonania:**

```python
1. Import modułów
   ├─> from src.modeling.neat import NEAT
   ├─> from src.environment import Environment
   └─> from src.logger import FitnessLogger, SpeciesLogger

2. Parsowanie argumentów
   └─> args.model_file = None (brak argumentu)

3. Inicjalizacja środowiska
   ├─> ENV_NAME = os.getenv("ENV_NAME", "MountainCarContinuous-v0")
   └─> env = Environment(ENV_NAME, seed=2137)

4. Utworzenie algorytmu NEAT
   ├─> input_size, output_size = env.get_params_num()
   └─> neat = NEAT(input_size, output_size)

5. Inicjalizacja loggerów
   ├─> fit_log = FitnessLogger()
   └─> spec_log = SpeciesLogger()

6. Trening
   └─> neat.train(
           env.run_env,                    # Funkcja ewaluacji
           weight_mutation_rate=0.3,       # 30% szansa mutacji wagi
           mutation_range=0.5,             # Zakres mutacji: [-0.5, 0.5]
           add_node_rate=0.01,             # 1% szansa dodania węzła
           add_connection_rate=0.01,       # 1% szansa dodania połączenia
           compatibility_threshold=3,      # Próg specjacji
           c1=0.5,                         # Waga excess genes
           c2=2,                           # Waga disjoint genes
           c3=2,                           # Waga różnicy wag
           best_individuals_copied=0.1,    # 10% elity
           num_generations=100,            # 100 generacji
           population_size=2000,           # 2000 osobników
           act=sigmoid,                    # Funkcja aktywacji
           verbose=True,                   # Wyświetlaj postęp
           callbacks=[fit_log, spec_log]   # Loggery
       )

7. Zapisanie wyników
   ├─> fit_log.save_chart_data("fitness_chart.png")
   ├─> spec_log.save_to_csv("species.csv")
   └─> best = neat.get_best()

8. Zapis modelu
   └─> pickle.dump(best, 
           f"models/{ENV_NAME}_{timestamp}.pkl")

9. Demonstracja
   ├─> reward, frames = env.run_env(best, store_gif=True)
   └─> store_episode_as_gif(frames, filename="gif.gif")
```

#### 2. Tryb ewaluacji (z argumentem)

```bash
python3 -m example.main models/MountainCarContinuous-v0_20260114_190000.pkl
```

**Przepływ wykonania:**

```python
1. Parsowanie argumentów
   └─> args.model_file = "models/..."

2. Inicjalizacja środowiska
   └─> env = Environment(ENV_NAME, seed=2137)

3. Wczytanie modelu
   └─> with open(args.model_file, "rb") as f:
           best = pickle.load(f)

4. Demonstracja
   ├─> print(best)              # Wyświetl genom
   ├─> print(best.nn)           # Wizualizacja sieci
   ├─> print(best.fitness)      # Fitness
   ├─> reward, frames = env.run_env(best, store_gif=True)
   ├─> print("reward:", reward)
   └─> store_episode_as_gif(frames, filename="gif.gif")
```

### Szczegóły parametrów treningu

| Parametr | Wartość | Opis |
|----------|---------|------|
| `weight_mutation_rate` | 0.3 | Prawdopodobieństwo mutacji każdej wagi |
| `mutation_range` | 0.5 | Maksymalna zmiana wagi podczas mutacji |
| `add_node_rate` | 0.01 | Prawdopodobieństwo dodania nowego węzła |
| `add_connection_rate` | 0.01 | Prawdopodobieństwo dodania nowego połączenia |
| `compatibility_threshold` | 3 | Próg dystansu genetycznego dla specjacji |
| `c1` | 0.5 | Waga genów excess w obliczaniu dystansu |
| `c2` | 2 | Waga genów disjoint w obliczaniu dystansu |
| `c3` | 2 | Waga różnicy wag w obliczaniu dystansu |
| `best_individuals_copied` | 0.1 | Procent najlepszych osobników kopiowanych bez zmian |
| `num_generations` | 100 | Liczba generacji ewolucji |
| `population_size` | 2000 | Liczba osobników w populacji |

### Pliki wyjściowe

Po zakończeniu treningu generowane są następujące pliki:

1. **`fitness_chart.png`**: Wykres fitness w funkcji generacji
2. **`species.csv`**: Historia liczby gatunków (format CSV z separatorem `;`)
3. **`models/{ENV_NAME}_{timestamp}.pkl`**: Zapisany najlepszy genom
4. **`gif.gif`**: Animacja demonstracji najlepszego modelu

---

## Diagram zależności

### Zależności między klasami

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│  (Orkiestracja treningu i ewaluacji)                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──────────────┐
             ↓              ↓
    ┌────────────────┐  ┌──────────────┐
    │  Environment   │  │    NEAT      │
    │                │  │              │
    │  - env         │  │  - population│
    │  - seed        │  │  - species   │
    └────────┬───────┘  └──────┬───────┘
             │                 │
             │                 ├─────────────┐
             │                 ↓             ↓
             │          ┌──────────┐   ┌──────────┐
             │          │  Genome  │   │  Logger  │
             │          │          │   │          │
             │          │  - nn    │   │  (ABC)   │
             │          │  - genes │   └────┬─────┘
             │          └────┬─────┘        │
             │               │              ├──────────┐
             │               ↓              ↓          ↓
             │          ┌─────────┐  ┌──────────┐ ┌──────────┐
             │          │   NN    │  │ Fitness  │ │ Species  │
             │          │         │  │ Logger   │ │ Logger   │
             │          │ - nodes │  └──────────┘ └──────────┘
             │          │ - conns │
             │          └────┬────┘
             │               │
             │               ├──────────────┐
             │               ↓              ↓
             │          ┌─────────┐   ┌──────────────┐
             │          │  Node   │   │  Connection  │
             │          │         │   │              │
             │          │ - index │   │ - from_node  │
             │          │ - type  │   │ - to_node    │
             │          │ - layer │   │ - weight     │
             │          │ - act   │   │ - innovation │
             │          └────┬────┘   └──────────────┘
             │               │
             │               ↓
             │          ┌──────────────┐
             │          │  activation  │
             │          │              │
             │          │  - sigmoid   │
             │          │  - tanh      │
             │          │  - identity  │
             │          └──────────────┘
             │
             ↓
    ┌─────────────────┐
    │   Gymnasium     │
    │   (external)    │
    └─────────────────┘
```

### Zależności importów

```
example/main.py
    ├─> src.modeling.neat (NEAT)
    ├─> src.modeling.activation (sigmoid, tanh)
    ├─> src.environment (Environment, store_episode_as_gif)
    ├─> src.config (ENV_NAME)
    └─> src.logger (FitnessLogger, SpeciesLogger)

src/modeling/neat.py
    ├─> src.modeling.genome (Genome)
    ├─> src.modeling.nn (NN)
    └─> src.modeling.activation (sigmoid)

src/modeling/genome.py
    ├─> src.modeling.nn (NN)
    └─> src.modeling.node (Connection)

src/modeling/nn.py
    ├─> src.modeling.node (Node, NodeTypes)
    └─> src.modeling.activation (sigmoid, identity)

src/modeling/node.py
    └─> (brak zależności wewnętrznych)

src/modeling/activation.py
    └─> numpy

src/environment.py
    ├─> gymnasium
    ├─> numpy
    └─> matplotlib

src/logger.py
    └─> matplotlib

src/config.py
    └─> python-dotenv
```

### Hierarchia dziedziczenia

```
Logger (ABC)
    ├─> FitnessLogger
    └─> SpeciesLogger

Enum
    ├─> NodeTypes
    └─> Environment.ActionType
```

---

## Konfiguracja i parametry

### Zmienne środowiskowe (`.env`)

```bash
# Nazwa środowiska Gymnasium
ENV_NAME=MountainCarContinuous-v0
```

### Obsługiwane środowiska

Implementacja obsługuje środowiska Gymnasium z następującymi typami przestrzeni akcji:

#### Przestrzeń dyskretna (`gym.spaces.Discrete`)
- **CartPole-v1**
- **Acrobot-v1**
- **MountainCar-v0**
- Inne środowiska z dyskretną przestrzenią akcji

#### Przestrzeń ciągła (`gym.spaces.Box`)
- **MountainCarContinuous-v0** (domyślne)
- **Pendulum-v1**
- Inne środowiska z ciągłą przestrzenią akcji

### Instalacja i zależności

#### Instalacja:

```bash
# Utworzenie środowiska wirtualnego
python3 -m venv .venv
source .venv/bin/activate

# Instalacja zależności
pip install -r requirements.txt
```

### Parametry algorytmu NEAT

#### Parametry mutacji strukturalnej

- **`add_node_rate`**: Kontroluje częstotliwość dodawania nowych neuronów
  - Wartości typowe: 0.001 - 0.05
  - Wyższa wartość = bardziej złożone sieci
  
- **`add_connection_rate`**: Kontroluje częstotliwość dodawania nowych połączeń
  - Wartości typowe: 0.001 - 0.1
  - Wyższa wartość = gęstsze sieci

#### Parametry mutacji wag

- **`weight_mutation_rate`**: Prawdopodobieństwo mutacji każdej wagi
  - Wartości typowe: 0.1 - 0.5
  - Wyższa wartość = szybsza eksploracja, mniejsza stabilność

- **`mutation_range`**: Maksymalna zmiana wagi
  - Wartości typowe: 0.1 - 1.0
  - Wyższa wartość = większe skoki w przestrzeni rozwiązań

#### Parametry specjacji

- **`compatibility_threshold`**: Próg dystansu genetycznego
  - Wartości typowe: 1.0 - 5.0
  - Niższa wartość = więcej gatunków (większa różnorodność)
  - Wyższa wartość = mniej gatunków (szybsza konwergencja)

- **`c1`, `c2`, `c3`**: Współczynniki dystansu genetycznego
  - `c1`: Waga genów excess (typowo: 0.5 - 2.0)
  - `c2`: Waga genów disjoint (typowo: 0.5 - 2.0)
  - `c3`: Waga różnicy wag (typowo: 0.1 - 3.0)
  - Wyższe wartości zwiększają znaczenie danego komponentu

#### Parametry reprodukcji

- **`best_individuals_copied`**: Procent elity
  - Wartości typowe: 0.05 - 0.2
  - Wyższa wartość = silniejszy elityzm (mniejsza różnorodność)

- **`population_size`**: Rozmiar populacji
  - Wartości typowe: 100 - 5000
  - Większa populacja = lepsza eksploracja, wolniejsze wykonanie

- **`num_generations`**: Liczba generacji
  - Wartości typowe: 50 - 500
  - Zależy od złożoności problemu
