import os
import pathlib

import pygad.torchga
import torch

import learning.entry_point as ga
from config import GeneralConfig
from learning.selection.crossover_type import CrossoverType
from learning.selection.mutation_type import MutationType
from learning.selection.parent_selection_type import ParentSelectionType


def main():
    def load_vector_from_file(file: pathlib.Path):
        model = GeneralConfig.EXPORT_MODEL_INIT()
        model.load_state_dict(torch.load(file), strict=False)
        vector = pygad.torchga.model_weights_as_vector(model)
        return vector

    dir_name = pathlib.Path("./report_external/epochs_best/epoch_1")

    initial_population = [
        load_vector_from_file(dir_name / model_file)
        for model_file in os.listdir(dir_name)
    ]

    ga.run(
        server_count=GeneralConfig.SERVER_COUNT,
        num_generations=GeneralConfig.NUM_GENERATIONS,
        num_parents_mating=GeneralConfig.NUM_PARENTS_MATING,
        keep_elitism=GeneralConfig.KEEP_ELITISM,
        parent_selection_type=ParentSelectionType.STOCHASTIC_UNIVERSAL_SELECTION,
        crossover_type=CrossoverType.UNIFORM,
        mutation_type=MutationType.RANDOM,
        mutation_percent_genes=GeneralConfig.MUTATION_PERCENT_GENES,
        initial_population=initial_population,
    )


if __name__ == "__main__":
    main()
