class StateMachine:

  def __init__(self):
    self.handlers = {}
    self.start_state = None
    self.end_states = []

  def add_state(self, name, handler, end_state=0):
    name = name.upper()
    self.handlers[name] = handler
    if end_state:
      self.end_states.append(name)

  def set_start(self, name):
    self.start_state = name.upper()

  def run(self, cargo):
    try:
      handler = self.handlers[self.start_state]
    except:
      raise RuntimeError("must call .set_start() before .run()")
    if not self.end_states:
      raise RuntimeError("at least one state must be an end_state")

    while True:
      (newState, cargo) = handler(cargo)
      if newState.upper() in self.end_states:
        print("reached ", newState)
        break 
      else:
        handler = self.handlers[newState.upper()]
