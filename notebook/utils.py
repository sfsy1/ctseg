import warnings
from collections import Counter


def check_multiple_cts(paths):
    grandparent_paths = [path.parent.parent for path in paths]
    counter = Counter(grandparent_paths)
    for grandparent, count in counter.items():
        if count > 1:
            print(
                "======= !!WARNING!! =======\n"
                f"Multiple CTs found for subject: {grandparent.stem}\n"
                f"Folder path: {grandparent}\n"
                "Check spreadsheet in data folder to for the relevant CT. Delete the rest."
            )
