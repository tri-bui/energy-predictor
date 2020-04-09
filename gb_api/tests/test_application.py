import pytest
import json

from api import application, config, __version__ as api_version
from gb_model import __version__ as model_version


@pytest.fixture
def app_instance():
    app = application.create_app(config.TestingConfig)
    with app.app_context():
        yield app


@pytest.fixture
def app_test_client(app_instance):
    with app_instance.test_client() as client:
        yield client


def test_health(app_test_client):
    response = app_test_client.get('/health')
    assert response.status_code == 200


def test_version(app_test_client):
    response = app_test_client.get('/version')
    response_json = json.loads(response.data)
    assert response.status_code == 200
    assert response_json['model_version'] == model_version
    assert response_json['api_version'] == api_version