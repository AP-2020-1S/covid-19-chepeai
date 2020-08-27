from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from data import df, cases_by_province
import os

loader = FileSystemLoader("templates")
env = Environment(loader=loader, autoescape=select_autoescape(['html', 'xml']))

template_files = [
    ('index', {"data": df, "cases_by_province": cases_by_province}),
    ('about', dict())
]

for t, data in template_files:
    template = env.get_template(f'{t}.jinja')

    output = template.render(**data)

    output_file = os.path.join("docs", f"{t}.html")

    with open(output_file, "w") as f:
        f.write(output)
