from abc import ABC, abstractmethod

# experiment interface
class Run(ABC):
  @abstractmethod
  def load_modules(self):
    pass

  @abstractmethod
  def fit(self, X, y):
    pass

  @abstractmethod
  def partial_fit(self, X):
    pass

  @abstractmethod
  def stop(self):
    pass


