personDict = dict()
personDict["name"] = "Alex"
personDict["age"] = 18
personDict["course"] = 1

for key in personDict:
    print(personDict[key])

print("---------")
personDict = {"name": "Alex", "age":18, "course": 1}

for key in personDict:
    print(personDict[key])

print("---------")

for val in personDict.values():
    print(val)

print("---------")

personDict = dict(name = "Alex", age = 18, cource = 1)

for key in personDict:
    print(personDict[key])

print("---------")