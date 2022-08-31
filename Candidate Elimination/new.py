import numpy as np
import pandas as pd

data = pd.read_csv('datenew.csv')
concepts = np.array(data.iloc[:, 0:-1])
print("Instances are:", concepts)
target = np.array(data.iloc[:, -1])
print("Target Values are: ", target)


def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and genearal_h")
    print("Specific Boundary: ", specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("Generic Boundary: ", general_h)

    for i, h in enumerate(concepts):
        print("Instance", i + 1, "is ", h)
        if target[i] == "yes":
            print("Instance is Positive ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == "no":
            print("Instance is Negative ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("Specific Bundary after ", i + 1, "Instance is ", specific_h)
        print("Generic Boundary after ", i + 1, "Instance is ", general_h)

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print("Final Specific_h: ", s_final)
print("Final General_h: ", g_final)