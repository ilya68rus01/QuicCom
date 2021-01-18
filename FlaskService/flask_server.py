from FlaskService import app, api
from flask_restful import Resource, reqparse
from PredictorService.NewNeuralNet import *
from PredictorService.PredictorService import *


def start():
	global neuralNet
	neuralNet = NewNeuralNet()
	neuralNet.prepare_data()
	api.add_resource(ODQA, "/get_answer", "/get_answer/")
	app.run(debug=True, host='0.0.0.0', port=8000)


class ODQA(Resource):
	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('comment')
		parser.add_argument('text')
		args = parser.parse_args()
		comment = str(args['comment']).lower()
		text = str(args['text']).lower()
		# predictor.batch_train_ann_model(comment)
		answer = neuralNet.predict_words(text)
		if answer == "" or answer == None:
			answer = ""
		print(answer)
		return {"answer": answer}, 200
