import ai_edge_litert
print(ai_edge_litert)
try:
    from ai_edge_litert import interpreter
    print('interpreter_module', interpreter)
    print('has_Interpreter', hasattr(interpreter, 'Interpreter'))
except Exception as exc:
    print(type(exc).__name__, exc)
