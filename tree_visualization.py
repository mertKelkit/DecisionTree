import graphviz
import datetime

from tree import *


generated_randoms = []


def draw_tree(tree, file_name=None, colnames=None, target_description=None, file_format='pdf', view=False):
    root = tree.root
    if file_name is None:
        file_name = 'tree_{date:%d-%m-%Y_%H_%M_%S}.gv'.format(date=datetime.datetime.now())
    g = graphviz.Digraph('Tree', filename=file_name, node_attr={'shape': 'note', 'style': 'filled',
                                                                'color': 'black',
                                                                'fontname': 'courier-bold',
                                                                'fontsize': '16.0',
                                                                'margin': '0.5, 0.5'},
                         edge_attr={'fontname': 'helvetica-bold'}, format=file_format)
    draw(root, g, colnames, target_description)
    for i, color in zip(range(len(root.targets)), Node.target_colors):
        hex = '#%02x%02x%02xb0' % color
        if target_description is not None:
            g.node(name=hex, fillcolor=hex, label='Target: ' + str(root.targets[i]) + '\n' +
                                                  str(target_description[root.targets[i]]),
                                                  _attributes={'margin': '0.6, 0.2'})
        else:
            g.node(name=hex, fillcolor=hex,
                   label='Target: ' + str(root.targets[i]), _attributes={'margin': '0.6, 0.2'})
    g.render(filename=file_name, view=view)
    return g


def draw(node, graph, colnames=None, target_description=None):

    global generated_randoms

    # Prevent duplications
    random.seed()
    node_id = random.randint(-1000000, 1000000)
    while node_id in generated_randoms:
        random.seed()
        node_id = random.randint(-1000000, 1000000)
    generated_randoms.append(node_id)

    if isinstance(node, TerminalNode):
        return
    if target_description is not None:
        target_label = '[' + ', '.join(str(target) + '- ' + str(target_description[target]) for target
                                       in node.targets) + ']'
    else:
        target_label = node.targets

    # draw node
    if colnames is None:
        if node.split_point is not None:
            tail_data = 'Referencing column is X[{}]\n' \
                        'Split points {}\n\n' \
                        'Targets {}\n' \
                        'Training Samples {}'.format(node.col_idx, node.split_point, target_label, node.frequencies)
        else:
            tail_data = 'Referencing column is X[{}]\n' \
                        'Targets {}\n' \
                        'Training Samples {}'.format(node.col_idx, target_label, node.frequencies)

    else:
        if node.split_point is not None:
            # split_point = [round(i, 2) for i in node.split_point]
            tail_data = 'Referencing column is {}\n' \
                        'Split points {}\n\n' \
                        'Targets {}\n' \
                        'Training Samples {}'.format(colnames[node.col_idx], node.split_point, target_label, node.frequencies)
        else:
            tail_data = 'Referencing column is {}\n' \
                        'Targets {}\n' \
                        'Training Samples {}'.format(colnames[node.col_idx], target_label, node.frequencies)

    graph.node(tail_data, tail_data, fillcolor=node.hex)

    for c, n in sorted(node.children.items()):

        if target_description is not None:
            target_label = '[' + ', '.join(str(target) + '- ' + str(target_description[target]) for target
                                           in n.targets) + ']'
        else:
            target_label = n.targets

        if isinstance(n, TerminalNode):
            result = []
            for p, t in zip(n.frequencies, n.targets):
                if p == 0:
                    continue
                result.append('\n{}% -> {}'.format(round((p/sum(n.frequencies))*100, 2), t))
            result = ''.join(s for s in result)

            head_data = 'Resulting target {}\n' \
                        '\nTargets {}\n' \
                        'Training Samples {}\n' \
                        'id = {}'.format(str(result), target_label, n.frequencies, node_id)
        else:
            if colnames is None:
                if n.split_point is not None:
                    head_data = 'Referencing column is X[{}]\n' \
                                'Split points {}\n\n' \
                                'Targets {}\n' \
                                'Training Samples {}'.format(n.col_idx, n.split_point, target_label, n.frequencies)
                else:
                    head_data = 'Referencing column is X[{}]\n' \
                                'Targets {}\n' \
                                'Training Samples {}'.format(n.col_idx, target_label, n.frequencies)
            else:
                if n.split_point is not None:
                    head_data = 'Referencing column is {}\n' \
                                'Split points {}\n\n' \
                                'Targets {}\n' \
                                'Training Samples {}'.format(colnames[n.col_idx], n.split_point, target_label, n.frequencies)
                else:
                    head_data = 'Referencing column is {}\n' \
                                'Targets {}\n' \
                                'Training Samples {}'.format(colnames[n.col_idx], target_label, n.frequencies)
        label = '  ' + str(c)
        graph.node(head_data, head_data.split('id = ')[0], fillcolor=n.hex)
        graph.edge(tail_data, head_data, label=label)
        draw(n, graph, colnames=colnames, target_description=target_description)
