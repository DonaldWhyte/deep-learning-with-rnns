import argparse
import uuid
from daft import Node, PGM
from matplotlib import rc


class Helpers:

    def add_label(self,
                  graph: PGM,
                  text: str,
                  x: float,
                  y: float,
                  label_id: str=None):
        if label_id is None:
            label_id = str(uuid.uuid4())
        graph.add_node(Node(
            label_id, text, x, y,
            plot_params={
                'fill': False,
                'linewidth': 0.0
            },
            label_params={
                'size': 'small'
            }))


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--graph_module',
        required=True,
        help='name of Python module that contains the generate_graph() '
             'function used to generate the desired graph')
    parser.add_argument(
        '-o', '--output_file',
        default='out',
        help='filename of generated graphs (without file extension)')
    args = parser.parse_args()

    rc("font", family="sans-serif", size=12)
    #rc('text', usetex=True)
    helpers = Helpers()

    graphs = __import__(args.graph_module).generate_graphs(helpers)
    for i in range(len(graphs)):
        graphs[i].render().figure.savefig(
            f'{args.output_file}_{i}.svg', dpi=250)


if __name__ == '__main__':
    _main()
