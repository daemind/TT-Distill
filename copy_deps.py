# ruff: noqa
import modulefinder
import os
import shutil

finder = modulefinder.ModuleFinder(path=[os.path.abspath("src")])
finder.run_script("demos/main.py")

src_root = os.path.abspath("src")
target_root = os.path.abspath("TT-Distill/src")

count = 0
for mod in finder.modules.values():
    mod_file = getattr(mod, "__file__", None)
    if mod_file and mod_file.startswith(src_root):
        rel_path = os.path.relpath(mod_file, src_root)
        target_path = os.path.join(target_root, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(mod_file, target_path)
        count += 1
