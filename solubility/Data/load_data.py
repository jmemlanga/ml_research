import os
from tdc.single_pred import ADME

# Local disk on linux-12 — NOT the quota'd network home.
# Falls back to the env var if you've set TDC_DATA_DIR, else the local path.

DATA_DIR = os.environ.get('TDC_DATA_DIR', '/tmp/jmema20/tdc-data')

data = ADME(name='Solubility_AqSolDB', path=DATA_DIR)

split = data.get_split()          # random split by default (70/10/20)
train, valid, test = split['train'], split['valid'], split['test']

print(f"train: {train.shape}, valid: {valid.shape}, test: {test.shape}")
print(train.columns.tolist())     # ['Drug_ID', 'Drug', 'Y']
print(train.head())

