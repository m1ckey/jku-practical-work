import json
import glob

results = glob.glob('results/*/*.json')
print(len(results))

for result in results:
    with open(result) as f:
        j = json.load(f)
        if len(j) != 284:
            print(f'{result:80}: {len(j)}')
