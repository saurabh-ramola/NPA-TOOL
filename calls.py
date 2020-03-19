from subprocess import call
call(["pip", "install","virtualenv"])
call(["source","envname/bin/activate"])
call(["python","app.py"])