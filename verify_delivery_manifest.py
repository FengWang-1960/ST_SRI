#!/usr/bin/env python3
"""Read-only SHA-256 and byte-size validation of this delivery's fixed inventory."""
from pathlib import Path
import csv, hashlib, json, sys
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'MANIFEST_SHA256.csv').open(encoding='utf-8',newline='')))
bad=[]; listed=set()
for row in rows:
 relative=Path(row['path'])
 if relative.is_absolute() or '..' in relative.parts:
  bad.append({'path':row['path'],'reason':'unsafe relative path'});continue
 p=root/relative;listed.add(row['path'])
 if not p.is_file() or p.is_symlink():
  bad.append({'path':row['path'],'reason':'missing file or symbolic link'});continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 if p.stat().st_size!=int(row['bytes']) or h.hexdigest()!=row['sha256']:
  bad.append({'path':row['path'],'reason':'size or SHA-256 mismatch'})
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='.DS_Store' and '__pycache__' not in p.parts}
extra=sorted(actual-listed-{'MANIFEST_SHA256.csv'})
result={'all_passed':not bad and not extra,'listed_files':len(rows),'total_files':len(actual),'bad':bad,'unlisted_files':extra,'scope':'File identity only; this command does not reexecute scientific analysis, inference or training.'}
print(json.dumps(result,indent=2));sys.exit(0 if result['all_passed'] else 1)
