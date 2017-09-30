import uuid

from daft import Node, PGM
from matplotlib import rc

rc("font", family="sans-serif", size=12)
#rc('text', usetex=True)

pgm = PGM([3.6, 3.5], origin=[0.7, 0])

def add_label(text, x, y, label_id=None):
    if label_id is None:
        label_id = str(uuid.uuid4())
    pgm.add_node(Node(
        label_id, text, x, y,
        plot_params={
            'fill': False,
            'linewidth': 0.0
        },
        label_params={
            'size': 'small'
        }))


pgm.add_node(Node('feature_0', '$x_0$', 1, 2))
pgm.add_node(Node('feature_1', '$x_1$', 1, 1))
pgm.add_node(Node('bias', '$b$', 2, 0.5))
pgm.add_node(Node('weighted_sum', 'Σ', 2, 1.5, observed=True))
pgm.add_node(Node('activation_function', '$f(x)$', 3, 1.5, observed=True))
pgm.add_node(Node('output', '$y$', 4, 1.5))

pgm.add_edge('feature_0', 'weighted_sum')
pgm.add_edge('feature_1', 'weighted_sum')
pgm.add_edge('bias', 'weighted_sum')
pgm.add_edge('weighted_sum', 'activation_function')
pgm.add_edge('activation_function', 'output')

add_label('$w_0$', 1.5, 2)
add_label('$w_1$', 1.5, 1)
add_label('1', 2.2, 1)

pgm.render()
pgm.render().figure.savefig('out.svg', dpi=150)
