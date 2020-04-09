import os
import pathlib


ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_PATH / 'logs'
LOG_PATH.mkdir(exist_ok=True)


class Config:
    DEVELOPMENT = False
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SERVER_PORT = 5000
    SECRET_KEY = 'secretkey'


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    SERVER_PORT = os.environ.get('PORT', 5000)


# class Config:
#     def __init__(self):
#         self.DEVELOPMENT = False
#         self.DEBUG = False
#         self.TESTING = False
#         self.CSRF_ENABLED = True
#         self.SERVER_PORT = 5000
#         self.SECRET_KEY = 'secretkey'
#
#
# class DevelopmentConfig(Config):
#     def __init__(self):
#         super().__init__()
#         self.DEVELOPMENT = True
#         self.DEBUG = True
#
#
# class TestingConfig(Config):
#     def __init__(self):
#         super().__init__()
#         self.TESTING = True
#
#
# class ProductionConfig(Config):
#     def __init__(self):
#         super().__init__()
#         self.SERVER_PORT = os.environ.get('PORT', 5000)
