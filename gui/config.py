import random


class Config:
    SEED = 422
    random.seed(SEED)


# Running Locally
class DevelopmentConfig(Config):
    #######################################################################
    # HUMAN3.6 ANNOTATIONS SETUP
    HUMAN_ANNOTATION_FILE = (
        '/Users/Robert/Documents/Caltech/CS81_Depth_Research/datasets/'
        'human36m_annotations/human36m_train_17.json')
    HUMAN_IMAGES_SERVER_FOLDER = ('/static/images/human_17/')


# Running Remotely
class ProductionConfig(Config):
    #######################################################################
    # HUMAN3.6 ANNOTATIONS SETUP
    HUMAN_ANNOTATION_FILE = ('/home/ubuntu/datasets/human3.6/annotations/'
                             'human36m_train_17.json')
    HUMAN_IMAGES_SERVER_FOLDER = '/static/images/human_17/'
