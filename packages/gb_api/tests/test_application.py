import pytest
import json
from gb_api.api import application, config
from gb_api import __version__ as api_version
from gb_model import __version__ as model_version


@pytest.fixture
def app_instance():

    """ Create an application with testing configuration. """

    app = application.create_app(config.TestingConfig)
    with app.app_context():
        yield app


@pytest.fixture
def app_test_client(app_instance):

    """ Get the test client from the application instance. """

    with app_instance.test_client() as client:
        yield client


def test_index(app_test_client):

    """ Test the status of the "/" route. """

    response = app_test_client.get('/')
    assert response.status_code == 200


def test_health(app_test_client):

    """ Test the status of the "/health" route. """

    response = app_test_client.get('/health')
    assert response.status_code == 200


def test_version(app_test_client):

    """ Test the status of the "/version" route. It should return the model and 
    API versions. """

    response = app_test_client.get('/version')
    response_json = json.loads(response.data)
    assert response.status_code == 200
    assert response_json['model_version'] == model_version
    assert response_json['api_version'] == api_version


def test_pred(app_test_client):

    """ Test the status of the "/v1/predict" route. It should return the model 
    version and predictions for the posted data. """

    # Load meter and weather data
    DATA_PATH = config.ROOT_PATH.parent / 'tests' / 'datasets'
    data = {}
    with open(DATA_PATH / 'm_application.json', 'r') as m:
        data['meter'] = json.load(m)
    with open(DATA_PATH / 'w_application.json', 'r') as w:
        data['weather'] = json.load(w)

    # Post data to API
    response = app_test_client.post('/v1/predict', json=data)
    response_json = json.loads(response.data)
    preds = response_json.get('predictions')
    ver = response_json.get('version')

    assert response.status_code == 200
    assert ver == model_version
    assert len(preds) == 96
