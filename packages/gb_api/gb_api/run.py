from gb_api.api import application, config


app = application.create_app(config.ProductionConfig)


if __name__ == '__main__':
    app.run(ssl_context='adhoc')
