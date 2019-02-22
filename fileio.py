import glob

def load_directory(path):
    # all directory names in path are class names
    # all files inside a directory share label
    class_paths = glob.glob(path + '/*')
    class_names = list(map(lambda x: os.path.split(x)[-1], class_paths))
    file_names = {x: glob.glob(os.path.join(path,x,'*')) for x in class_names}
    return class_names, file_names


def pickle_results(path,file_name,data):
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(os.path.join(path,file_name),'w+')
    pickle.dump(data,f)
    f.close()
    return True


def file_dict_to_flat(file_dict):
    file_list = []
    for class_name in file_dict:
        file_list.extend( file_dict[class_name])
    return file_list


def file_list_to_dict(file_list):
    file_dict = {}
    for f in file_list:
        class_name = f.split('/')[-2]
        if class_name in file_dict:
            file_dict[class_name].append(f)
        else:
            file_dict[class_name] = [f]
    return file_dict
