from datetime import datetime

_logDir = None

def getLogDir():
  global _logDir
  if _logDir is None:
    _logDir = 'log/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
  return _logDir
