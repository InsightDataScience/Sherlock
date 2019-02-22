def load_directory(path):
    # all directory names in path are class names
    # all files inside a directory share label
    class_paths = glob.glob(path + '/*')
    class_names = list(map(lambda x: os.path.split(x)[-1], class_paths))
    file_names = {x: glob.glob(os.path.join(path,x,'*')) for x in class_names}
    return class_names, file_names
