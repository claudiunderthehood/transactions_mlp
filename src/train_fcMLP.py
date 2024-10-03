import datetime

import torch
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler

from generator_modules.load_dataset import load_ds

from classification_modules.normaliser import normalize_data
from classification_modules.classificationClass import Classification

from deepLearning_modules.seeder import seed_everything
from deepLearning_modules.fcMLPClass import SimpleTransactionMLP
from deepLearning_modules.training_tools import prepare_data_loaders
from deepLearning_modules.training_tools import train_model_earlystopping_metrics
from deepLearning_modules.fcMLPDropoutClass import SimpleTransactionMLPWithDropout

SEED: int = 42

PATH: str = './data/transformed/'

START: str = '2024-06-11'
END: str = '2024-09-14'

print("Reading files...")
transactions_df: pd.DataFrame = load_ds(PATH, START, END)
print("Transactions read from files: ", len(transactions_df))
print("Fraudulent transactions: ", transactions_df.IS_FRAUD.sum())

if torch.cuda.is_available():
    device: str = "cuda" 
else:
    device = "cpu"
print("Selected device is",device)

model_name: str = 'best_fc.pth'
model_name2: str = 'best_fc_dropout.pth'

columns_to_exclude: list = ['TRX_ID', 'CLIENT_ID', 'TERMINAL_ID', 'IS_FRAUD', 'FRAUD_SCENARIO', 'TRX_DATETIME', 'TRX_SECONDS', 'TRX_DAYS']

input_features: list = [col for col in transactions_df.columns if col not in columns_to_exclude]
output_features: str = 'IS_FRAUD'

start_date_training: datetime = datetime.datetime.strptime("2024-07-25", "%Y-%m-%d")

classificate: Classification = Classification()

training_duration=7
delay_duration=7
test_duration=7

delta_valid = test_duration

start_date_training_with_valid = start_date_training+datetime.timedelta(days=-(delay_duration+delta_valid))

(train_set, valid_set)=classificate.split_train_test(transactions_df,start_date_training_with_valid,
                                       training_duration=training_duration,delay_duration=delay_duration,test_duration=test_duration)

(train_set, valid_set)=normalize_data(train_set, valid_set, input_features)

trainer, validator = prepare_data_loaders(train_df=train_set, valid_df=valid_set, input_features=input_features, output_feature=output_features, batch_size=64, device=device)

seed_everything(SEED)

model = SimpleTransactionMLP(len(input_features)).to(device)

trainer, validator = prepare_data_loaders(train_df=train_set, valid_df=valid_set, input_features=input_features, output_feature=output_features, batch_size=64, device=device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

results = train_model_earlystopping_metrics(model=model, trainer=trainer, validator=validator, optimizer=optimizer, criterion=criterion, patience=3, max_epochs=500, verbose=True, model_name=model_name, scheduler=scheduler)

seed_everything(SEED)

model = SimpleTransactionMLPWithDropout(len(input_features), 0.2).to(device)

trainer, validator = prepare_data_loaders(train_df=train_set, valid_df=valid_set, input_features=input_features, output_feature=output_features, batch_size=64, device=device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

results = train_model_earlystopping_metrics(model=model, trainer=trainer, validator=validator, optimizer=optimizer, criterion=criterion, patience=3, max_epochs=500, verbose=True, model_name=model_name2, scheduler=scheduler)