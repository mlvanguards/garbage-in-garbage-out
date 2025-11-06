from abc import ABC, abstractmethod


class BaseMetadataExtractor(ABC):

    @abstractmethod
    def extract(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
