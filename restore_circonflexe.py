# -*- coding: utf-8 -*-
"""Restaure les « â » supprimés par erreur, à partir d'un commit de référence.

Usage (PowerShell, dans le dossier du projet) :
    py restore_circonflexe.py 07d8a25 app_preprocess.py auto_preprocessor.py test_auto_preprocessor.py
    py restore_circonflexe.py 07d8a25 app_preprocess.py auto_preprocessor.py test_auto_preprocessor.py --apply

Sans --apply : simulation (rien n'est écrit). Seules les lignes qui ne diffèrent
de l'ancienne version que par « â » → « a » sont restaurées ; les autres sont listées.
"""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def key(line: str) -> str:
    return line.strip().replace("â", "a")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv
    if len(args) < 2:
        print(__doc__)
        return 1
    ref, files = args[0], args[1:]
    total_fixed = 0
    for name in files:
        res = subprocess.run(["git", "show", f"{ref}:{name}"], capture_output=True)
        if res.returncode != 0:
            print(f"[{name}] introuvable dans {ref} : {res.stderr.decode('utf-8', 'replace').strip()}")
            continue
        old_lines = res.stdout.decode("utf-8").splitlines()
        path = Path(name)
        cur = path.read_bytes().decode("utf-8").splitlines(keepends=True)

        old_by_key = defaultdict(list)
        for ln in old_lines:
            old_by_key[key(ln)].append(ln)
        cur_by_key = defaultdict(list)
        for i, ln in enumerate(cur):
            cur_by_key[key(ln)].append(i)

        fixed, manual = 0, []
        for k, olds in old_by_key.items():
            if not any("â" in o for o in olds) or not k:
                continue
            idxs = cur_by_key.get(k, [])
            if len(idxs) == len(olds):
                for i, o in zip(idxs, olds):
                    eol = cur[i][len(cur[i].rstrip("\r\n")):]
                    if cur[i].rstrip("\r\n") != o:
                        cur[i] = o + eol
                        fixed += 1
            else:
                manual.extend(o for o in olds if "â" in o)
        total_fixed += fixed
        print(f"[{name}] lignes restaurées : {fixed} ; à traiter à la main : {len(manual)}")
        for o in manual:
            print(f"    ancienne ligne : {o.strip()[:120]}")
        if apply and fixed:
            path.write_bytes("".join(cur).encode("utf-8"))
    print("TERMINÉ (écrit)" if apply else "SIMULATION : relancer avec --apply pour écrire")
    return 0


if __name__ == "__main__":
    sys.exit(main())