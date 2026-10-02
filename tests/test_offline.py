import csv

import eval.offline as offline


def test_append_csv_row_lines_up_with_an_existing_header(tmp_path):
    path = tmp_path / "summary.csv"
    offline.append_csv_row(path, {"name": "a", "correct": 1, "wall_s": 9.5})
    # a later script writes its columns in a different order and adds one
    offline.append_csv_row(path, {"name": "b", "dev_correct": 3, "correct": 2, "wall_s": 4.0})

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0] == {"name": "a", "correct": "1", "wall_s": "9.5", "dev_correct": ""}
    assert rows[1] == {"name": "b", "correct": "2", "wall_s": "4.0", "dev_correct": "3"}
