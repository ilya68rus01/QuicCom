from FlaskService import app, api
from flask_restful import Resource, reqparse
from PredictorService.PredictorService import *


def start():
	global predictor
	predictor = PredictorService()
	predictor.__create_w2v_model__()
	predictor.__load_ann__()
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
		predictor.batch_train_ann_model(comment)
		answer = predictor.predict_next_word(text)
		if answer == "" or answer == None:
			answer = ""
		print(answer)
		return {"answer": answer}, 200
