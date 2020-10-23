from FlaskService import app, api
from flask_restful import Resource, reqparse


def start():
	api.add_resource(ODQA, "/get_answer", "/get_answer/")
	app.run(debug=True, host='0.0.0.0', port=80)


class ODQA(Resource):
	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('question')
		args = parser.parse_args()
		question = args['question']
		print(question)
		answer = question + "awdasfse"
		if answer == "" or answer == None:
			answer = "Затрудняюсь ответить"
		print(answer)
		return {"answer": answer}, 200