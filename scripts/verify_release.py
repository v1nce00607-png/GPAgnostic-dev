"""Verify OPT1 source manifest and model-relative candidate status integrity."""
from pathlib import Path
import argparse,json,hashlib,re
from grasp_planner.models import verdict
from grasp_planner.geometry_planner import GATES

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'artifact_manifest_OPT1.json')
    a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.results.resolve();errors=[]
    if a.manifest.exists():
        for r in json.loads(a.manifest.read_text(encoding='utf-8'))['files']:
            path=(root/r['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:
                errors.append('hash '+r['path'])
    else:errors.append('missing OPT1 manifest')
    file=out/'all_candidates.json'
    if not file.exists():errors.append(f'missing results: {file}');rows=[]
    else:rows=json.loads(file.read_text(encoding='utf-8'))
    for r in rows:
        if r['status']!=verdict(r['checks'],GATES) or r.get('executable',False):errors.append('status '+r['id'])
    for p in out.glob('*.html'):
        for link in re.findall(r'(?:src|href)=[\"\']([^\"\']+)[\"\']',p.read_text(encoding='utf-8')):
            if any(link.startswith(x) for x in ['http:','https:','data:','#','mailto:']):continue
            if '\\' in link or '<' in link or '>' in link:continue
            if not (p.parent/link.split('#')[0]).exists():errors.append(f'link {p.name}: {link}')
    print(json.dumps(dict(candidates=len(rows),errors=errors),indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
