import json
from flask import Flask, Blueprint, request, jsonify
from gb_api.api import logging_config
from gb_api import __version__ as api_version
from gb_model import predict, __version__ as model_version


logger = logging_config.get_logger(__name__) # logger
pred_app = Blueprint('pred_app', __name__) # app blueprint


# /
@pred_app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        health_route = '<a href="/health">health</a>'
        version_route = '<a href="/version">version</a>'
        return health_route + '<br/>' + version_route


# /health
@pred_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        logger.info('health status OK')
        return 'ok'


# /version
@pred_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        logger.info(
            f'model_version: {model_version} - api_version: {api_version}'
        )
        return jsonify(
            {'model_version': model_version, 'api_version': api_version}
        )


# /v1/predict
@pred_app.route('/v1/predict', methods=['POST'])
def pred():
    if request.method == 'POST':

        # Input files
        input_files = request.files
        logger.info(f'Number of files: {len(input_files)}\n'
                    f'Input files: {input_files}')

        # Load json from input files
        if len(input_files) > 0:
            meter_data = json.load(input_files['meter'])
            weather_data = json.load(input_files['weather'])

        # Get json from request if there are no input files
        else:
            input_data = request.get_json()
            meter_data = input_data.get('meter')
            weather_data = input_data.get('weather')

        # Log input data
        logger.info(f'Input meter data: {meter_data}\n'
                    f'Input weather data: {weather_data}')

        # Make predictions
        output = predict.make_prediction(meter_data, weather_data)
        logger.info(f'Output: {output}')

        # Predictions and version
        predictions = output.get('predictions')
        ver = output.get('version')
        return jsonify({'predictions': predictions, 'version': ver})


# Create app
def create_app(config_object) -> Flask:
    app = Flask('gb_api')
    app.config.from_object(config_object)
    app.register_blueprint(pred_app)
    logger.info('Application created.')
    return app
