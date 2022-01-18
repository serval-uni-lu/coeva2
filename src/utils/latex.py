import os
import subprocess


def apply_standalone_template(latex_str):

    template = r"""
    \documentclass[border=3mm]{{standalone}}
    \usepackage{{booktabs}}
    \usepackage{{amssymb}}
    \begin{{document}}
    {0}
    \end{{document}}
    """
    return template.format(latex_str)


def export_latex(latex_str, path):

    cwd = os.getcwd()
    directory_path, file_path = os.path.split(path)

    os.chdir(directory_path)
    with open(file_path, "w") as f:
        f.write(apply_standalone_template(latex_str))
        
    with open(os.devnull, "w") as f:
        subprocess.call(["pdflatex", file_path], stdout=f)

    # subprocess.run(["pdflatex", file_path])
    os.chdir(cwd)
