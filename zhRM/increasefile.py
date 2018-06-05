increasedfile = open("zhENVRILarge.txt","w",encoding="utf-8")
originalfile = open("zhENVRI.txt","r",encoding="utf-8")
content = originalfile.read()
for i in range(1,500):
       increasedfile.write(content)
