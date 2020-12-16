from datetime import datetime


def PrintDatetime(msg="Done"):
    print("{}: {}".format(msg, datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
