from flask import Flask, Blueprint, request, jsonify

from api import logging_config, __version__ as api_version
from gb_model import predict, __version__ as model_version


logger = logging_config.get_logger(__name__)

pred_app = Blueprint('pred_app', __name__)


@pred_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        logger.info('health status OK')
        return 'ok'


@pred_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version, 'api_version': api_version})


@pred_app.route('/v1/predict', methods=['POST'])
def pred():
    if request.method == 'POST':
        input_data = request.get_json()

        meter_data = input_data.get('meter')
        logger.info(f'Input meter data: {meter_data}')

        weather_data = input_data.get('weather')
        logger.info(f'Input weather data: {weather_data}')

        output = predict.make_prediction(meter_data, weather_data)
        logger.info(f'Output: {output}')

        predictions = output.get('predictions')
        ver = output.get('version')

        return jsonify({'predictions': predictions, 'version': ver})


def create_app(config_object) -> Flask:
    app = Flask('gb_api')
    app.config.from_object(config_object)
    app.register_blueprint(pred_app)
    logger.info('Application created.')
    return app
