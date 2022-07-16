'''
Checks if the type of answer can be predicted based on question structure
'''

class QuestionTypeMatcher:
    def __init__(self, question):
        self.question = question

    def get_type(self,question):

        date_type_question_beginnings = ['in what year', 'in which year', 'when was the', 'what year was', 'what year did','on what date']

        if ' '.join(question.split()[:3]) in date_type_question_beginnings:
            return True,'DATE'

        else:
            return False,'DATE'