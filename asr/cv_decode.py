import requests, csv, json, os
from tqdm import tqdm

# Transcribe a file using asr api
def transcribe(filename):
    transcription = ""
    files = {'file': open(filename, 'rb')}
    r = requests.post('http://127.0.0.1:8001/asr', files=files)
    obj = json.loads(r.text)
    transcription = obj['transcription']
    return transcription

# Transcribe all files in csv, write into updated csv
outfile = open('out.csv', 'w')
writer = csv.writer(outfile)

with open('cv-valid-dev.csv') as csvfile:
    rows = list(csv.reader(csvfile))
    rows[0].append('generated_text')
    writer.writerow(rows[0])
    print(f"Transcribing {len(rows)-1} files...")
    for row in tqdm(rows[1:]):
        row.append(transcribe(row[0]))
        writer.writerow(row)

outfile.close()

os.replace('out.csv', 'cv-valid-dev.csv')
