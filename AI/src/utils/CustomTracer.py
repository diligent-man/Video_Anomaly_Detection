"""
Workaround for limitation of symbolic tracing from pytorch
Ref:
    1/ https://pytorch.org/docs/stable/fx.html
    2/ https://github.com/pytorch/pytorch/issues/51803
"""
import torch


__all__ = ["CustomTracer"]


class CustomTracer(torch.fx.Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """

    def __init__(self, *args, custom_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_leaf_module = custom_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about

            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.custom_leaf_module and isinstance(m, self.custom_leaf_module):
            return True

        # Added part
        if hasattr(m, "is_leaf_module") and m.is_leaf_module:
            return True

        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)
