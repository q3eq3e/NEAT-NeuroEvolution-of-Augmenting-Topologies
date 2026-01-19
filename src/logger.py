from matplotlib import pyplot as plt


class Logger:
    def log(self, generation: int, stats: dict) -> None:
        pass


class FitnessLogger(Logger):
    def __init__(self) -> None:
        self.records = []

    def log(self, generation: int, stats: dict) -> None:
        self.records.append(stats["fitness"])

    def get_logs(self) -> list[float]:
        return self.records

    def save_chart_data(self, filepath: str) -> None:
        plt.plot(self.records)
        plt.title("Fitness over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.savefig(filepath)


class SpeciesLogger(Logger):
    def __init__(self) -> None:
        self.records = []

    def log(self, generation: int, stats: dict) -> None:
        self.records.append(stats["species"])

    def get_logs(self) -> list[list[int]]:
        return self.records

    def save_to_csv(self, filepath: str) -> None:
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(self.records)
