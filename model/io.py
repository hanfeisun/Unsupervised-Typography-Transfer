import scipy.misc


def dump_image(filename, data):
    print("Save to %s" % filename)
    scipy.misc.imsave(filename, data.reshape([64, 64]))
