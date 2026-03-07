# ruff: noqa
import modulefinder
import os
import shutil

finder = modulefinder.ModuleFinder(path=[os.path.abspath('src')])
finder.run_script('demos/main.py')

src_root = os.path.abspath('src')
target_root = os.path.abspath('TT-Distill/src')

count = 0
for mod in finder.modules.values():
    if mod.__file__ and mod.__file__.startswith(src_root):
        rel_path = os.path.relpath(mod.__file__, src_root)
        target_path = os.path.join(target_root, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(mod.__file__, target_path)
        count += 1
