import json
with open('info.json','r') as f:
    info = json.load(f)
for field in info['data_desc']['fields']:
    col = field['column']
    desc = field['properties'].get('description','')
    if 'accepted' in desc or 'denied' in desc:
        print(col, ':', desc)
