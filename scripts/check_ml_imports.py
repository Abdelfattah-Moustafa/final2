from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp

print('litert_interpreter', Interpreter)
print('mp_module', mp)
print('mp_version', getattr(mp, '__version__', None))
print('solutions', hasattr(mp, 'solutions'))
print('holistic', hasattr(mp.solutions, 'holistic') if hasattr(mp, 'solutions') else False)
