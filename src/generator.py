import pandas as pd

from generator_modules.save_dataset import save_ds
from generator_modules.generatorClass import Generator

generate: Generator = Generator(num_clients=5000, num_terminals=10000, seed_clients=0, seed_terminals=0)

transactions_df: pd.DataFrame = generate.dataset(num_days=183, radius=5)

save_ds(transactions_df, './data/csv/', './data/raw/')