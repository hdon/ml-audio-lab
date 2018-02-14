def iterWindow(iterable, windowSize):
  window = []
  for item in iterable:
    window.append(item)
    if len(window) < windowSize:
      continue
    yield tuple(window)
    window = window[1:]
