from typing import Annotated,get_origin
from types import GenericAlias
import inspect

_TOOL_HOOKS={}
_TOOL_DESCRIPTIONS={}    

def register_tool(func:callable):
    tool_params=[]
    tool_name=func.__name__
    tool_description=inspect.getdoc(func)
    func_params=inspect.signature(func).parameters
    for name, param in func_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter `{name}` missing type annotation")
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")   
        type,(description,required)=annotation.__origin__,annotation.__metadatea__
        type:str =str(type) if isinstance(type,GenericAlias)  else type.__name__
          