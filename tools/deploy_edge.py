"""Copy tracked Edge scripts into the existing board workspace, with backups.

Default is dry-run. Does not install packages, modify OS settings, or copy models.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(repository, target):
    mappings = []
    for source in sorted((repository / 'edge/darknet53').glob('*.py')):
        mappings.append((source, target / 'darknet53' / source.name))
    for source in sorted((repository / 'edge/network_test').glob('*.py')):
        mappings.append((source, target / 'network_test' / source.name))
    for source in sorted((repository / 'edge/fcm_tools').glob('*.py')):
        subdir = 'fcm_l37' if source.name in ('verify_live_fcm.py','benchmark_fcm_encode.py') else 'fcm_l37/fcm_bundle'
        mappings.append((source, target / subdir / source.name))
    if not mappings:
        raise ValueError('No Edge source scripts found')
    return mappings


def deploy(repository, target, apply=False):
    repository = repository.resolve()
    target = target.expanduser().resolve()
    mappings = plan(repository, target)
    changes = []
    for source, dest in mappings:
        if dest.is_symlink() or (dest.exists() and not dest.is_file()):
            raise ValueError(f'Refusing non-regular destination: {dest}')
        # Reject redirected parent directories before writing any files.
        if not dest.resolve().is_relative_to(target):
            raise ValueError(f'Destination escapes target workspace: {dest}')
        if dest.exists() and sha(dest) == sha(source):
            continue
        changes.append((source,dest))
    if not apply:
        for source,dest in changes:
            print(f'[dry-run] {source.relative_to(repository)} -> {dest}')
        print(f'{len(changes)} file(s) would change. Add --apply to copy with backups.')
        return dict(applied=False,changed=len(changes))
    if not changes:
        print('All scripts already match; no files changed.')
        return dict(applied=True,changed=0)
    backup = target / '.edge-backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup.mkdir(parents=True)
    records = []
    # Complete every old-file backup before replacing any destination.
    for source,dest in changes:
        relative = dest.relative_to(target)
        previous = sha(dest) if dest.exists() else None
        if previous:
            saved = backup / relative
            saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(dest,saved)
        records.append(dict(path=str(relative),previous_sha256=previous,new_sha256=sha(source)))
    manifest = dict(status='backed_up',target=str(target),files=records)
    manifest_path = backup / 'deploy_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    for source,dest in changes:
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    manifest['status'] = 'complete'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(f'Copied {len(changes)} script(s). Backup and manifest: {backup}')
    return dict(applied=True,changed=len(changes),backup=str(backup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root',type=Path,default=Path.home()/'yolox_project',
                        help='Copy target only; runtime still expects ~/yolox_project')
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    deploy(Path(__file__).resolve().parents[1],args.target_root,args.apply)


if __name__=='__main__':
    main()
