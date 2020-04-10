from api import application, config


app = application.create_app(config.TestingConfig)


if __name__ == '__main__':
    app.run()
