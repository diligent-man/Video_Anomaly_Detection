"""
This module is identical to torchvision.models.feature_extraction import create_feature_extractor
but used with custom LeafModuleAwareTracer.
"""
import re

from collections import OrderedDict
from typing import Callable

import torch

from .LeafModuleAwareTracer import LeafModuleAwareTracer


__all__ = ["NodePathTracer"]


class NodePathTracer(LeafModuleAwareTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the
    name of the Node from which the operation originated. A node name here is
    a `.` separated path walking the hierarchy from top level module down to
    leaf operation or leaf module. The name of the top level module is not
    included as part of the node name. For example, if we trace a module whose
    forward method applies a ReLU module, the name for that node will simply
    be 'relu'.

    Some notes on the specifics:
        - Nodes are recorded to `self.node_to_qualname` which is a dictionary
          mapping a given Node object to its node name.
        - Nodes are recorded in the order which they are executed during
          tracing.
        - When a duplicate node name is encountered, a suffix of the form
          _{int} is added. The counter starts from 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track the qualified name of the Node being traced
        self.current_module_qualname = ""
        # A map from FX Node to the qualified name\#
        # NOTE: This is loosely like the "qualified name" mentioned in the
        # torch.fx docs https://pytorch.org/docs/stable/fx.html but adapted
        # for the purposes of the torchvision feature extractor
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of `fx.Tracer.call_module`
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Adds the qualified name of the caller to
           `current_module_qualname` for retrieval by `create_proxy`
        3) Once a leaf module is reached, calls `create_proxy`
        4) Restores the caller's qualified name into current_module_qualname
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy("call_module", module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname

    def create_proxy(
        self, kind: str, target: torch.fx.node.Target, args, kwargs, name=None, type_expr=None, *_
    ) -> torch.fx.proxy.Proxy:
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(self, module_qualname: str, node: torch.fx.node.Node) -> str:
        node_qualname = module_qualname

        if node.op != "call_module":
            # In this case module_qualname from torch.fx doesn't go all the
            # way to the leaf function/op, so we need to append it
            if len(node_qualname) > 0:
                # Only append '.' if we are deeper than the top level module
                node_qualname += "."
            node_qualname += str(node)

        # Now we need to add an _{index} postfix on any repeated node names
        # For modules we do this from scratch
        # But for anything else, torch.fx already has a globally scoped
        # _{index} postfix. But we want it locally (relative to direct parent)
        # scoped. So first we need to undo the torch.fx postfix
        if re.match(r".+_[0-9]+$", node_qualname) is not None:
            node_qualname = node_qualname.rsplit("_", 1)[0]

        # ... and now we add on our own postfix
        for existing_qualname in reversed(self.node_to_qualname.values()):
            # Check to see if existing_qualname is of the form
            # {node_qualname} or {node_qualname}_{int}
            if re.match(rf"{node_qualname}(_[0-9]+)?$", existing_qualname) is not None:
                postfix = existing_qualname.replace(node_qualname, "")
                if len(postfix):
                    # existing_qualname is of the form {node_qualname}_{int}
                    next_index = int(postfix[1:]) + 1
                else:
                    # existing_qualname is of the form {node_qualname}
                    next_index = 1
                node_qualname += f"_{next_index}"
                break
        return node_qualname
