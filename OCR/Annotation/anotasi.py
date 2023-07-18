from typing import Dict, List
import jiwer
import os


def merge_lines(file_paths: List[str]):
    line_dict = dict()
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = " ".join([line.replace("\n", "") for line in file.readlines()])
            line_dict[str(file_path).replace("Annotations/", "")] = lines
    return line_dict


def check_agreement(dict: Dict[str, str], sets: str):
    IRFAN = [(key, dict.get(key)) for key in [x for x in dict.keys() if x.startswith(f"{sets}_Irfan")]]
    KENTA = [(key, dict.get(key)) for key in [x for x in dict.keys() if x.startswith(f"{sets}_kenta")]]
    RAFI = [(key, dict.get(key)) for key in [x for x in dict.keys() if x.startswith(f"{sets}_Rafi")]]

    for i in range(len(IRFAN)):
        I = IRFAN[i][1]
        K = KENTA[i][1]
        R = RAFI[i][1]

        IK = jiwer.process_words(I, K)
        IR = jiwer.process_words(I, R)
        KR = jiwer.process_words(K, R)

        result = f"""=============RESULT=============

IRFAN VS KENTA
{jiwer.visualize_alignment(IK, show_measures=False)}Substitutions: {IK.substitutions}
Insertions   : {IK.insertions}
Deletions    : {IK.deletions}
Hits         : {IK.hits} of {len(I.split()) if len(I.split()) > len(K.split()) else len(K.split())}


IRFAN VS RAFI
{jiwer.visualize_alignment(IR, show_measures=False)}Substitutions: {IR.substitutions}
Insertions   : {IR.insertions}
Deletions    : {IR.deletions}
Hits         : {IR.hits} of {len(I.split()) if len(I.split()) > len(R.split()) else len(R.split())}


KENTA VS RAFI
{jiwer.visualize_alignment(KR, show_measures=False)}Substitutions: {KR.substitutions}
Insertions   : {KR.insertions}
Deletions    : {KR.deletions}
Hits         : {KR.hits} of {len(K.split()) if len(K.split()) > len(R.split()) else len(R.split())}

SHEET -> {f"{IRFAN[i][0][-9:-4]}.txt"};{IK.substitutions};{IK.insertions};{IK.deletions};{IK.hits};{len(I.split()) if len(I.split()) > len(K.split()) else len(K.split())};{IR.substitutions};{IR.insertions};{IR.deletions};{IR.hits};{len(I.split()) if len(I.split()) > len(R.split()) else len(R.split())};{KR.substitutions};{KR.insertions};{KR.deletions};{KR.hits};{len(K.split()) if len(K.split()) > len(R.split()) else len(R.split())}
""".replace("sentence 1\n", "")

        with open(f"Agreement/{sets}/{IRFAN[i][0][-9:-4]}.txt", "w", encoding="utf-8") as file:
            print(result, file=file)


sets = [
    "basdat_4", "basdat_10", "jarkom_6", "jarkom_8", "jarkom_9"
]
annotations = dict()

for set_name in sets:
    paths = [f'{set_name}_{x}' for x in ["kenta", "Irfan", "Rafi"]]
    for path in paths:
        file_paths = list(map(lambda file: f"Annotations/{path}/{file}", os.listdir(f"Annotations/{path}")))
        annotations.update(merge_lines(file_paths))
    check_agreement(annotations, set_name)
