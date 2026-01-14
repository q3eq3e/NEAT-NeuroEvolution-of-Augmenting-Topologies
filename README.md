# Dokumentacja projektu NEAT

Autorzy: Jakub Bagiński, Maciej Borkowski, Patryk Skręta

## Szybki start

Polecenia należy wykonywać z poziomu katalogu głównego projektu.

### Trening nowego modelu
```bash
python3 -m example.main
```

### Ewaluacja zapisanego modelu
```bash
python3 -m example.main models/NAZWA_MODELU.pkl
```

### Konfiguracja środowiska - wybór jednego z dostępnych środowisk Gymnasium
Stwórz plik `.env`z jednyną z poniższych linii, aby ustawić środowisko, na którym będzie trenowany model:
```bash
ENV_NAME=MountainCarContinuous-v0
```
```bash
ENV_NAME=MountainCar-v0
```
```bash
ENV_NAME=Acrobot-v1
```
```bash
ENV_NAME=CartPole-v1
```

## Szczegółowa dokumentacja

[README](./docs/README.md)

## Szczegóły zadania

[descritpion](./description.md)
