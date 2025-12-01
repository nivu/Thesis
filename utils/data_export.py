import csv

class CSVExporter:
    def __init__(self, filename, header):
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self._write_header(header)

    def _write_header(self, header):
        self.csv_writer.writerow(header)

    def write_row(self, row_data):
        self.csv_writer.writerow(row_data)

    def close(self):
        self.csv_file.close()