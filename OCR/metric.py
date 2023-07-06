from typing import List
import jiwer


def wer(ref: str | List[str], hyp: str | List[str]):
    return jiwer.wer(ref, hyp)


def cer(ref: str | List[str], hyp: str | List[str]):
    return jiwer.cer(ref, hyp)
