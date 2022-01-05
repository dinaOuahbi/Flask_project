from flask import abort
class PostClass:

    POST = [
        {'id':1, 'title':'first_Post', 'Content':'This is my first post'},
        {'id':2, 'title':'seconde_Post', 'Content':'This is my seconde post'},
        {'id':3, 'title':'third_Post', 'Content':'This is my third post'},
        {'id':4, 'title':'forty_Post', 'Content':'This id my fourty post'},
        {'id':5, 'title':'fifty_Post', 'Content':'This id my fifty post'}
    ]

    @classmethod
    def all(cls):
        #fetch all posts
        return cls.POST
        

    @classmethod
    def find(cls, id):
        # fetch one post
        try:
            return cls.POST[int(id)- 1]
        except IndexError:
            abort(404)


class Client:

    CLIENTS = [
        {'id':1, 'name':'toto', 'achats':10},
        {'id':2, 'name':'tato', 'achats':11},
        {'id':3, 'name':'tito', 'achats':12},
        {'id':4, 'name':'tbto', 'achats':4},
        {'id':5, 'name':'tota', 'achats':1},
        {'id':6, 'name':'htoto', 'achats':6},
        {'id':7, 'name':'htota', 'achats':20}
    ] 

    @classmethod
    def all(cls):
        #fetch all posts
        return cls.CLIENTS

    @classmethod
    def find(cls, id):
        # fetch one post
        try:
            return cls.CLIENTS[int(id)- 1]
        except IndexError:
            abort(404)