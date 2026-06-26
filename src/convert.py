#helper py file only for converting submission file formats
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
df.to_excel(sys.argv[2], index=False)