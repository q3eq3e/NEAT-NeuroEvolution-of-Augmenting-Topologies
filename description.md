# Temat 9: Neuro-ewolucja: neuro-evolution of augmented topologies (NEAT) 
Neuro Ewolucja jest metodą pozwalającą znajdować małe sieci które są dostosowane do konkretnych rozwiązań. Najczęściej są to sieci które są o wiele mniejsze i to co potocznie jest wykorzystywane w MLP. Klasycznym algorytmem w tej przestrzeni, jest algorytm NEAT, który ewoluuje sieci neuronów na grafie. Jest to idealny przykład inżynierii inspirowanej biologią.

### Wymagania:
- Wytrenowanie i przetestowanie sieci NEAT na 3 środowiskach z gymnasium (przykładowo cart, pendulum i acrobot).
- Zbadaj jakie architektury sieci ewoluują dla każdego ze środowisk (grafy które produkuje sieć),

### Na plus:
- Sieć neat pozwala każdemu neuronowi posiadać inną funkcję aktywacji, zbadaj jakie funkcje aktywacji są wykorzystywane po nauczeniu w każdym zadaniu (uważaj niektóre implementacje NEAT korzystają tylko z jednej funkcji aktywacji), 
- Sieci ewoluują zazwyczaj w mniejsze rozwiązania, jednak przetwarzanie grafów jest dla nich bootleneckiem. Z drugiej strony klasyczne sieci FeedForward są przetwarzane niezwykle szybko ze względu na uproszczenie ich obliczeń do mnożenia macierzy. Zastanów się w jaki sposób sieć grafowa może być zmieniona na sieć FeedForward.

### Dane:
- https://gymnasium.farama.org/environments/classic_control/cart_pole/
- https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
- https://gymnasium.farama.org/environments/classic_control/acrobot/

### Materiały:
- https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
- https://www.youtube.com/watch?v=ihX3-WDua2I
- https://github.com/PeterWaIIace/NEAT
- https://pyneat.readthedocs.io/en/latest/
