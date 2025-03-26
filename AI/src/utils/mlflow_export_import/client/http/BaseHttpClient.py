from abc import ABCMeta, abstractmethod

__all__ = ["BaseHttpClient"]


class BaseHttpClient(metaclass=ABCMeta):
    @abstractmethod
    def _get(self, resource, params=None):
        pass

    @abstractmethod
    def _put(self, resource, data=None):
        pass

    @abstractmethod
    def _post(self, resource, data=None):
        pass

    @abstractmethod
    def _delete(self, resource):
        pass

    @abstractmethod
    def _patch(self, resource, data=None):
        pass

    @abstractmethod
    def get(self, resource, params=None):
        pass

    @abstractmethod
    def put(self, resource, data=None):
        pass

    @abstractmethod
    def post(self, resource, data=None):
        pass

    @abstractmethod
    def delete(self, resource):
        pass

    @abstractmethod
    def patch(self, resource, data=None):
        pass

    @abstractmethod
    def get_api_uri(self):
        pass

    @abstractmethod
    def get_token(self):
        pass
