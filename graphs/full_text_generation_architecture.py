# Parameters
SEQUENCE_LENGTH = 30
NUM_HIDDEN_LAYERS = 3
UNROLLED = False

g = PGM([15, 10], origin=[0.0, 0.0])

# -----------------------------------------------------------------------------
# * Placeholders
# -----------------------------------------------------------------------------
input_layer = helpers.layer(
    g,
    [f'$x_{{{i}}}$' for i in range(98)],
    1, 4, spacing=0.75)

input_layer = g.add_node(Node('input_layer', '$x$', 1, 2))

# -----------------------------------------------------------------------------
# * Hidden Layers (the things we want to learn the weights of)
# -----------------------------------------------------------------------------
hidden_layers = [
    [Node(TODO)]
    for i in range(NUM_HIDDEN_LAYERS)
]
if UNROLLED:
    pass

else:
    hidden_layers = [
        helpers.layer(g)
        for i in range(SEQUENCE_LENGTH)
    ]


# -----------------------------------------------------------------------------
# * Outputs
# -----------------------------------------------------------------------------

input_layer = g.add_node(Node('input_layer', '$x$', 1, 2))

# -----------------------------------------------------------------------------
# * Plumbing/Reporting
# -----------------------------------------------------------------------------

# TODO: less important

