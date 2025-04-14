'''Simple utility functions.

'''
import csv


def write_csv(new_csv_file, rows_new, fieldnames=None):
    with open(new_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        if fieldnames is None:
            fieldnames = rows_new[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows_new)


def read_csv(csv_file_path, fieldnames=None):
    with open(csv_file_path, encoding='utf-8') as csv_file:
        rows = csv.DictReader(csv_file, fieldnames=fieldnames)
        rows = list(rows)
    return rows
