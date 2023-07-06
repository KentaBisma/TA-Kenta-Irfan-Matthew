from OCR import do_OCR, raw_OCR
import pandas as pd
import os


def prepare_output(inp: str):
    return " ".join(inp.split())


def export_result(df: pd.DataFrame, folder: str):
    df.to_csv(f"result/{folder}/result_{folder}.csv", sep="\t", index=False)


for folder in os.listdir("Dataset"):
    list = []
    for filename in os.listdir(f"Dataset/{folder}"):
        result = prepare_output(do_OCR(f"Dataset/{folder}/{filename}"))
        if (result == ""):
            continue
        print(result)
        list.append((filename.split(".")[0], result))
        print("------------------------------------")

    df = pd.DataFrame(list, columns=["timestamp", "text"])
    export_result(df, folder)

# for folder in os.listdir("Dataset"):
#     list_raw = []
#     for filename in os.listdir(f"Dataset/{folder}"):
#         result_raw = prepare_output(raw_OCR(f"Dataset/{folder}/{filename}"))
#         if (result_raw == ""):
#             continue
#         print(result_raw)
#         list_raw.append((filename.split(".")[0], result_raw))
#         print("------------------------------------")

#     df_raw = pd.DataFrame(list_raw, columns=["timestamp", "text"])
#     export_result(df_raw, f"raw_{folder}")
