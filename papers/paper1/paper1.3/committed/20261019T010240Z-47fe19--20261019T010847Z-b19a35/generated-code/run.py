import json,subprocess,sys
with open('/inputs/protocol.json') as f: p=json.load(f)
for name in p['models']:
    with open(name+'.log','w') as log:
        subprocess.run([sys.executable,'experiment.py','train','--model',name],stdout=log,stderr=subprocess.STDOUT,check=True)
