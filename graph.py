from graphviz import Digraph

from value import Value


def draw_dot(root, rankdir="LR", fmt="svg"):
    dot = Digraph(format=fmt, graph_attr={"rankdir": rankdir})
    for node in root.topo_sort():
        uid = str(id(node))
        dot.node(uid, shape="record",
                 label=f"{{ {node.label} | data {node.data:.4f} | grad {node.grad:.4f} }}")
        if node.op:
            dot.node(uid + node.op, label=node.op)
            dot.edge(uid + node.op, uid)
        for child in node.prev:
            dot.edge(str(id(child)), uid + node.op)
    return dot


if __name__ == "__main__":
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a*b
    e.label = "e"
    d = e + c
    d.label = "d"
    out = d.relu()
    out.label = "out"

    out.backward()
    draw_dot(out).render("graph", view=True)
