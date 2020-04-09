from api import application, config


app = application.create_app(config.DevelopmentConfig)


if __name__ == '__main__':
    app.run()
