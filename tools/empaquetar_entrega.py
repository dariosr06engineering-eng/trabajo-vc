"""Comprime la carpeta de entrega en entregas/team_G17.zip, con la estructura que pide el enunciado.

    python tools/empaquetar_entrega.py

El zip contiene la carpeta team_G17/ entera (README.md, requirements.txt, main.py, src/ y weights/), sin cachés de
Python. Antes de comprimir comprueba lo mismo que el paso 1 de check_submission.py (ficheros obligatorios, tamaño,
carpetas prohibidas y versiones exactas en requirements.txt).
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))
import check_submission  # noqa: E402

TEAM = "team_G17"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def main():
    team_dir = ROOT / TEAM
    files = [f for f in sorted(team_dir.rglob("*"))
             if f.is_file() and not SKIP_DIRS & set(f.relative_to(team_dir).parts) and f.suffix != ".pyc"]
    if not (team_dir / "weights" / "fruits.pt").is_file():
        sys.exit("Falta weights/fruits.pt: entrena el modelo o cópialo antes de empaquetar")

    out = ROOT / "entregas" / f"{TEAM}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path(TEAM) / f.relative_to(team_dir))

    # Misma comprobación de estructura que check_submission.py, sobre lo que hay dentro del zip
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        tmp = ROOT / "entregas" / "_comprobacion"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        z.extractall(tmp)
    errors, warnings, size = check_submission.check_folder(tmp / TEAM)
    import shutil
    shutil.rmtree(tmp)

    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB comprimido, {size:.1f} MB sin comprimir, {len(names)} ficheros):")
    for n in names:
        print("  ", n)
    for e in errors:
        print("ERROR:", e)
    for w in warnings:
        print("aviso:", w)
    if errors:
        sys.exit(1)
    print("Estructura correcta.")


if __name__ == "__main__":
    main()
