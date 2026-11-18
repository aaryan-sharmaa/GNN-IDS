# Data directory

Place the two NetFlow v3 datasets here (they are not tracked by git):

| File | Dataset | Size |
|---|---|---|
| `NF-UNSW-NB15-v3.csv` | NF-UNSW-NB15-v3 (2,365,424 flows, 55 columns) | ~550 MB |
| `NF-CSE-CIC-IDS2018-v3.csv` | NF-CSE-CIC-IDS2018-v3 (20,115,529 flows, 55 columns) | ~4.0 GB |

Both are published by the University of Queensland (Sarhan et al., *NetFlow Datasets for
Machine Learning-based Network Intrusion Detection Systems*). Download the CSV versions
from the NF-v3 collection: <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>.

Other locations work too: pass `--unsw-csv` and `--cse-csv` to `main.py`.
