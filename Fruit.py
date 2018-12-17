class Fruit:

  def __init__(self, track_window, hist, counter = 0, centers = []):
      self.track_window = track_window
      self.hist = hist
      self.counter = counter
      self.centers = centers
      self.is_falling = False