import pandas as pd
import os

for folder in os.listdir("Gold Standard"):
    list = []
    for filename in os.listdir(f"Gold Standard/{folder}"):
        if filename.split(".")[-1] == "txt":
            with open(f"Gold Standard/{folder}/{filename}", "r", encoding="UTF-8") as file:
                list.append((filename.split(".")[0], file.readline().replace("\n", "")))
    
    df = pd.DataFrame(list, columns=["timestamp", "text"])
    df.to_csv(f"Gold Standard/{folder}/{folder}.csv", sep="\t", index=False)
