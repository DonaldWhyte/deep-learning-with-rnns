import argparse
import math
from typing import List
import uuid

from daft import Node, PGM
from matplotlib import rc


class Helpers:

    def layer(self,
              graph: PGM,
              node_texts: List[str],
              x: float,
              y: float,
              spacing: float=1,
              spacing_pow: float=1,
              **other_node_params) -> List[Node]:
        nodes = [
            Node(
                str(uuid.uuid4()),
                node_texts[i],
                x,
                y - (spacing * float(i)),
                **other_node_params)
            for i in range(len(node_texts))]
        for node in nodes:
            graph.add_node(node)
        return nodes

    def fully_connect(self,
                      graph: PGM,
                      layer1: List[Node],
                      layer2: List[Node]):
        for l1_node in layer1:
            for l2_node in layer2:
                graph.add_edge(l1_node.name, l2_node.name)

    def add_label(self,
                  graph: PGM,
                  text: str,
                  x: float,
                  y: float,
                  label_id: str=None,
                  color: str=None,
                  size: str=None,
                  weight: str=None):
        if label_id is None:
            label_id = str(uuid.uuid4())
        graph.add_node(Node(
            label_id, text, x, y,
            plot_params={
                'fill': False,
                'linewidth': 0.0
            },
            label_params={
                'color': color or 'black',
                'size': size or 'small',
                'weight': weight or 'normal'
            }))

    def add_label_range(self,
                        graph: PGM,
                        labels: List[str],
                        x: float,
                        y: float,
                        spacing: float=1,
                        spacing_exp: float=1,
                        direction: str='H',  # Horizontal or Vertical
                        overrides: dict=None,
                        **other_label_args):
        n_labels = len(labels)
        if direction == 'H':
            coordinates = [(x + (spacing * float(i)), y) for i in range(n_labels)]
        elif direction == 'V':
            coordinates = [
                (x, y - math.pow(spacing * float(i), spacing_exp))
                for i in range(n_labels)
            ]
        else:
            raise ValueError(f'Invalid direction: {direction}')

        overrides = overrides or {}

        for i in range(n_labels):
            label_text = labels[i]
            label_args = dict(other_label_args)
            label_overrides = overrides.get(label_text, {})
            for key, value in label_overrides.items():
                label_args[key] = value
            self.add_label(
                graph, label_text, coordinates[i][0], coordinates[i][1],
                **label_args)


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
