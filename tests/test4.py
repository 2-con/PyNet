import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.structure import Node
from tools.visual import tree_display

root = Node(0)
split1 = Node(1)
split1.add_child(Node(2))
split1.add_child(Node(3))
root.add_child(split1)
root.add_child(Node(4))

tree_display(root)