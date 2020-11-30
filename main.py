import sys
from FlaskService import flask_server
from PredictorService.PredictorService import PredictorService


def main():
    predictor = PredictorService()
    predictor.convert_to_vec()
    #flask_server.start()


if __name__ == '__main__':
    sys.exit(main())