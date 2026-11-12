"""Validate publishable text files without printing possible secret values."""
import ast
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.pytest_cache','__pycache__','.venv','venv'}
LOCAL_ROOTS={'data','models','outputs','results','checkpoints'}
FORBIDDEN={'.safetensors','.pt','.pth','.bin','.parquet','.pem','.key','.sqlite','.zip','.gz','.log'}
PATTERNS={
    'private_key':r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',
    'hub_token':r'\bhf_[A-Za-z0-9]{25,}\b',
    'github_token':r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    'cloud_key':r'\bAKIA[0-9A-Z]{16}\b',
    'private_server':r'podtcp\.compshare\.cn',
    'user_absolute_path':r'[A-Z]:[\\/]+Users[\\/]+\w+',
    'password_literal':r'''(?i)\bpassword\s*=\s*["'][^"'\s]{8,}["']''',
    'signed_url':r'(?i)[?&](?:X-Amz-Signature|X-Amz-Credential)=',
}

def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and p.relative_to(ROOT).parts[0] not in LOCAL_ROOTS
                  and not any(x in IGNORED for x in p.relative_to(ROOT).parts))

def main():
    issues=[]; counted=0
    for path in files():
        relative=path.relative_to(ROOT).as_posix();counted+=1
        if path.suffix in FORBIDDEN or path.name.startswith('.env') or path.stat().st_size>5*1024**2:
            issues.append((relative,'blocked file type/size'));continue
        try: text=path.read_text(encoding='utf-8')
        except UnicodeError:issues.append((relative,'non-text payload'));continue
        for label,pattern in PATTERNS.items():
            if re.search(pattern,text):issues.append((relative,label))
        if path.suffix=='.py':
            try:ast.parse(text)
            except SyntaxError:issues.append((relative,'Python syntax'))
        if path.suffix=='.json':
            try:json.loads(text)
            except ValueError:issues.append((relative,'JSON syntax'))
        if path.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                local=target.split('#')[0]
                if local and not (path.parent/local).exists():issues.append((relative,'broken link: '+local))
    if issues:
        for file,reason in issues:print(file+': '+reason)
        raise SystemExit(1)
    print(f'PASS: {counted} publishable text files; syntax, relative links, size and credential-pattern checks.')
    print('Pattern checks are a release guard, not a guarantee that all possible secrets can be detected.')

if __name__=='__main__':main()
