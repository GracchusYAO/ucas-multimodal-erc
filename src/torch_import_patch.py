"""Small workaround for occasional torch import failures in the local WSL env."""

from __future__ import annotations

import inspect
import sys
import types
import builtins


COMMON_BUILTINS = {
    "bool": builtins.bool,
    "abs": builtins.abs,
    "all": builtins.all,
    "any": builtins.any,
    "callable": builtins.callable,
    "dict": builtins.dict,
    "enumerate": builtins.enumerate,
    "filter": builtins.filter,
    "float": builtins.float,
    "getattr": builtins.getattr,
    "hasattr": builtins.hasattr,
    "hash": builtins.hash,
    "id": builtins.id,
    "int": builtins.int,
    "iter": builtins.iter,
    "isinstance": builtins.isinstance,
    "issubclass": builtins.issubclass,
    "len": builtins.len,
    "list": builtins.list,
    "map": builtins.map,
    "max": builtins.max,
    "min": builtins.min,
    "next": builtins.next,
    "object": builtins.object,
    "property": builtins.property,
    "range": builtins.range,
    "reversed": builtins.reversed,
    "round": builtins.round,
    "set": builtins.set,
    "setattr": builtins.setattr,
    "sorted": builtins.sorted,
    "str": builtins.str,
    "sum": builtins.sum,
    "super": builtins.super,
    "tuple": builtins.tuple,
    "type": builtins.type,
    "zip": builtins.zip,
}


def restore_common_builtins() -> None:
    """恢复常用内置函数，绕过当前环境偶发的导入期污染。"""
    for name, value in COMMON_BUILTINS.items():
        setattr(builtins, name, value)


def patch_inspect_for_torch() -> None:
    """避免 torch 导入时因为 inspect 源码定位异常而直接崩掉。"""
    original_getmodule = inspect.getmodule
    original_getframeinfo = inspect.getframeinfo

    def safe_getmodule(object, _filename=None):
        try:
            return original_getmodule(object, _filename)
        except Exception:
            # torch.library 注册 fake op 时只把它用于定位调用方模块；失败时 None 可接受。
            return None

    def safe_getframeinfo(frame, context: int = 1):
        try:
            return original_getframeinfo(frame, context)
        except Exception:
            # PyTorch 注册 fake op 只需要一个可显示的源码位置；这里用占位值即可。
            return inspect.Traceback("<unknown>", 0, "<unknown>", None, None)

    inspect.getmodule = safe_getmodule
    inspect.getframeinfo = safe_getframeinfo


def stub_torch_dynamo(torch_module=None) -> None:
    """当前项目不使用 torch.compile；用轻量 stub 避免导入 torch._dynamo。"""
    dynamo = types.ModuleType("torch._dynamo")

    def disable(fn=None, *args, **kwargs):
        if fn is None:
            return lambda inner_fn: inner_fn
        return fn

    dynamo.disable = disable
    dynamo.allow_in_graph = disable
    dynamo.disallow_in_graph = disable
    dynamo.optimize = disable
    dynamo.graph_break = lambda *args, **kwargs: None
    dynamo.is_compiling = lambda: False
    dynamo.config = types.SimpleNamespace()
    dynamo.__path__ = []  # 让 import torch._dynamo.xxx 把它当 package 看待

    trace_module = types.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")
    utils_module = types.ModuleType("torch._dynamo.utils")

    class TransformGetItemToIndex:
        pass

    trace_module.TransformGetItemToIndex = TransformGetItemToIndex
    utils_module.is_compile_supported = lambda *args, **kwargs: False

    sys.modules.setdefault("torch._dynamo", dynamo)
    sys.modules.setdefault("torch._dynamo._trace_wrapped_higher_order_op", trace_module)
    sys.modules.setdefault("torch._dynamo.utils", utils_module)
    if torch_module is not None:
        torch_module._dynamo = dynamo

    try:
        import torch.utils._config_module as config_module

        # 当前 WSL/conda 环境里 inspect/tokenize 偶发异常；项目不依赖这些 compile 注释。
        config_module.get_assignments_with_compile_ignored_comments = lambda module: set()
    except Exception:
        pass
