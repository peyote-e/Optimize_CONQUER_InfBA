'''
Checks if the type of answer can be predicted based on question structure
'''

class QuestionTypeMatcher:
    def __init__(self, question):
        self.question = question

    def get_type(self,question):

        date_type_question_beginnings = ['in what year', 'in which year', 'when was the', 'what year was', 'what year did','on what date']

        gpe_type_question_beginnings = ['in which country', 'where is the', 'where was the', 'where is the', 'what country is', 'what city was', 'in which city']

        if ' '.join(question.split()[:3]) in date_type_question_beginnings:
            return True,'DATE'
        if ' '.join(question.split()[:3]) in gpe_type_question_beginnings:
            return True, 'GPE'

        else:
            return False,'DATE'