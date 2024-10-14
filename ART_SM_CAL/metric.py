import re

def check_answer_new(row):
    if row.prediction != "":
        answer = re.sub('[^A-Za-z0-9]+', '', row.answer.lower())
        prediction = re.sub('[^A-Za-z0-9]+', '', row.prediction.lower())
        return (answer in prediction) or (prediction in answer)
    return False

def check_answer(row):
    if row.prediction != "":
        answer = row.answer.lower()
        prediction = row.prediction.lower()
        return (answer in prediction) or (prediction in answer)
    return False
