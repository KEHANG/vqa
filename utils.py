
def log_to_file(msg):
    f = open('output.txt','a+')
    f.write(msg+'\n')
    f.close()