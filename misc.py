def train_test_validate(path, model_path, n_test=50, n_val=50,
                      names=['test', 'val', 'train'] ):
    class_name = os.path.split(path)[-1]
    file_names = glob.glob(os.path.join(path,'*'))
    destinations = map(lambda x: os.path.join(model_path, x,
                                              class_name), names)
    random.shuffle(file_names)
    for d in destinations:
        if not os.path.isdir(d):
            os.makedirs(d)
        
    map(lambda x: shutil.move(x, os.path.join(destinations[0],
                            os.path.split(x)[-1])), file_names[0:n_test] )
    map(lambda x: shutil.move(x, os.path.join(destinations[1],
                        os.path.split(x)[-1])), file_names[nTest:n_test + n_val] )
    map(lambda x: shutil.move(x, os.path.join(destinations[2],
                            os.path.split(x)[-1])), file_names[n_test + n_val:] )
