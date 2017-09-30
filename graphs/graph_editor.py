from io import StringIO
from urllib.parse import unquote

from flask import Flask, request, Response, send_from_directory

from generate_graph import Helpers


app = Flask(__name__)


@app.route('/static/<path:path>')
def static_file(path):
    return send_from_directory('static', path)


@app.route('/render')
def render():
    # Retrieve generation code and wrap in required boilerplate
    generation_code = request.args.get('code')
    if not generation_code:
        print('No generation code given')
        return Response('', mimetype='image/svg+xml')
    generation_code = unquote(generation_code)
    generation_code = generation_code.replace('\n', '\n    ')
    code_wrapper = f'''\
# Define objects and modules that generation code can use
helpers = Helpers()
from daft import PGM, Node

def generate():
    {generation_code}
    return g

graph = generate()'''

    # Run generation code supplied by the user
    print(f'Running code:\n{code_wrapper}')
    exec(code_wrapper, globals())

    # Render the resulting SVG file and return it
    svg_buffer = StringIO()
    graph.render().figure.savefig(svg_buffer, format='svg')
    svg = svg_buffer.getvalue()

    return Response(svg, mimetype='image/svg+xml')

