import sys
sys.path.append("..")
import url_request as urequest
from urllib.parse import quote
from sys import argv
from IMQ import IMessageQueue
import json
import tools.random_helper as random_helper

class Classify_CN():

    #def __init__(self, port_num):
    #    print('init cnn classify')
    #    self.url = "http://localhost:"+str(port_num)+"/"

    #def getCategory(self, sentence):
    #    url = self.url+"s/"+quote(sentence.replace(' ','|'))
    #    result = urequest.request_url(url)
    #    if result:
    #        #print(result)
    #        return result
    #    return {}

    mq = None  # message queue

    def __init__(self, field):
        self.field = field
        publish_key = 'nlp.classify.rnn.'+field+'.request.'+random_helper.random_string()
        receive_key = publish_key.replace('request', 'reply')
        self.mq = IMessageQueue(receive_key, publish_key, receive_key, receive_key, '')

    def getCategory(self, sentence):
        if not sentence:
            return {}

        if type(sentence) != list:
            sentence = [sentence]

        result = self.mq.request_synchronize_paralle(sentence)
        if result:
            if type(result) == list:
                # return [json.loads(r) for r in result]
                result_list = [json.loads(r) for r in result]
                max_prob = -1
                # idx = 0
                final_res = {}
                for res in result_list:
                    if res['probability'] > max_prob:
                        max_prob = res['probability']
                        final_res = res
                    else:
                        pass

                return final_res
            else:
                return json.loads(result)
        return {}
        #     # result = self.mq.request_synchronize(sentence)
        # if result:
        #     return json.loads(result)
        # return {}

    def getCategories(self, sentences):
        if not sentences:
            return []
        if type(sentences) != list:
            sentences = [sentences]
        result = self.mq.request_synchronize_paralle(sentences)
        if result:
            if type(result) == list:
                return [json.loads(r) for r in result]
            else:
                return [json.loads(result)]
        return []

if __name__ == '__main__':
    script, field = argv
    # field = 'ecovacs_field'
    cc = Classify_CN(field)
    while 1:
        s = input('input: ')
        # s = ['你好', '我 来 取款 你好']
        result = cc.getCategory(s)
        print(result)
