def brightenImg(train, test, val, factor):
    # Brighten images
    print "Creating brightened images..."
    for j in range (0, train.shape[0]):
        for i in range(0, train.shape[1]):
            if train[j][i] <= (255 - factor):
                train[j][i]     = train[j][i] + factor
            else:
                train[j][i]     = 255
    for j in range (0, test.shape[0]):
        for i in range(0, test.shape[1]):
            if test[j][i] <= (255 - factor):
                test[j][i]     = test[j][i] + factor
            else:
                test[j][i]     = 255
    for j in range (0, val.shape[0]):
        for i in range(0, val.shape[1]):
            if val[j][i] <= (255 - factor):
                val[j][i] = val[j][i] + factor
            else:
                val[j][i] = 255

    return train, test, val

def darkenImg(train, test, val, factor):
    # Darken images
    print "Creating darkened images..."
    for j in range (0, train.shape[0]):
        for i in range(0, train.shape[1]):
            if train[j][i] >= factor:
                train[j][i]     = train[j][i] - factor
            else:
                train[j][i]     = 0
    for j in range (0, test.shape[0]):
        for i in range(0, test.shape[1]):
            if test[j][i] >= factor:
                test[j][i]     = test[j][i] - factor
            else:
                test[j][i]     = 0
    for j in range (0, val.shape[0]):
        for i in range(0, val.shape[1]):
            if val[j][i] >= factor:
                val[j][i]     = val[j][i] - factor
            else:
                val[j][i] = 0

    return train, test, val
