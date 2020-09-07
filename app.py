from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from utils.dashboard import data, data_by_province
from utils.model import preds  
import os

loader = FileSystemLoader("templates")
env = Environment(loader=loader, autoescape=select_autoescape(['html', 'xml']))

template_files = [
    ('index', {"data": data, "data_by_province": data_by_province}),
    ('model', {"preds": preds}),
    ('about', dict()),
]

for t, data in template_files:
    template = env.get_template(f'{t}.jinja')

    output = template.render(**data)

    output_file = os.path.join("docs", f"{t}.html")

    with open(output_file, "w") as f:
        f.write(output)
