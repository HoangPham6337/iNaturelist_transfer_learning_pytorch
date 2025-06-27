import pandas as pd
import json
from typing import Dict

with open("./data/haute_garonne/species_composition.json") as data:
    species_composition: Dict[str, int] = json.load(data)

df = pd.DataFrame(list(species_composition.items()), columns=["Species", "Count"])
print(df.sort_values(by="Count", ascending=False).head(10))