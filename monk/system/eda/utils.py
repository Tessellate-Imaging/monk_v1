from system.eda.imports import *
from system.imports import *


@accepts(str, str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def get_img_label(fname, delimiter):
    '''
    Find list of images and corresponding labels in csv file

    Args:
        fname (str): Path to csv file
        delimiter (str): Delimiter for csv file

    Returns:
        list: List of images in dataset
        list: List of labels corresponding every image in dataset
    '''
    f = open(fname);
    lst = f.readlines();
    f.close();
    del lst[0]
    img_list = [];
    label_list = [];
    for i in range(len(lst)):
        img, label = lst[i][:len(lst[i])-1].split(delimiter);
        img_list.append(img);
        label_list.append(label);

    return img_list, label_list;



@accepts(str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def read_csv(fname):
    '''
    Read CSV File

    Args:
        fname (str): Path to csv file

    Returns:
        list: List of every row in csv
    '''
    f = open(fname);
    lst = f.readlines();
    f.close();
    del lst[0]
    return lst;




@accepts(str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_folder_train(tpath):
    '''
    Find number of images in every class image folder - train

    Args:
        tpath (str): Path to image training folder

    Returns:
        list: List of classes
        list: List of number of images in every class
    '''
    classes_folder = os.listdir(tpath);
    classes_folder_strength = [];
    for i in range(len(classes_folder)):
        classes_folder_strength.append(len(os.listdir(tpath + "/" + classes_folder[i])));
    return classes_folder, classes_folder_strength;




@accepts(str, str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_folder_trainval(tpath, vpath):
    '''
    Find number of images in every class image folder - train and val

    Args:
        tpath (str): Path to image training folder
        vpath (str): Path to image validation folder

    Returns:
        list: List of classes
        list: List of number of images in every class
    '''
    classes_folder, classes_folder_strength = populate_from_folder_train(tpath);
    for i in range(len(os.listdir(vpath))):
            classes_folder_strength[i] = classes_folder_strength[i] + len(os.listdir(vpath + "/" + classes_folder[i]));
    return classes_folder, classes_folder_strength;


@accepts(str, str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_csv_train(tpath, delimiter):
    '''
    Find number of images in every class image csv - train

    Args:
        tpath (str): Path to csv pointing to training dataset
        delimiter (str): Delimiter for csv file

    Returns:
        list: List of classes
        list: List of number of images in every class
    '''
    img_list, label_list = get_img_label(tpath, delimiter);
    classes_folder = list(np.unique(sorted(label_list)))
    classes_folder_strength = [];
    for i in range(len(classes_folder)):
        classes_folder_strength.append(label_list.count(classes_folder[i]));
    return classes_folder, classes_folder_strength;

@accepts(str, str, str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_csv_trainval(tpath, vpath, delimiter):
    '''
    Find number of images in every class image csv - train and val

    Args:
        tpath (str): Path to csv pointing to training dataset
        vpath (str): Path to csv pointing to validation dataset
        delimiter (str): Delimiter for csv file

    Returns:
        list: List of classes
        list: List of number of images in every class
    '''
    classes_folder, classes_folder_strength = populate_from_csv_train(tpath, delimiter);
    img_list, label_list = get_img_label(vpath, delimiter);
    for i in range(len(classes_folder)):
        classes_folder_strength[i] += label_list.count(classes_folder[i]);
    return classes_folder, classes_folder_strength;


@accepts(str, str, str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def populate_missing(tpath, dataset_path, delimiter):
    '''
    Find number of missing images in imageset

    Args:
        tpath (str): Path to csv pointing to training dataset
        dataset_path (str): Path to dataset containing images
        delimiter (str): Delimiter for csv file

    Returns:
        list: List of missing images
    '''
    lst = read_csv(tpath);
    missing_images = [];
    for i in range(len(lst)):
        img, label = lst[i][:len(lst[i])-1].split(delimiter);
        if(not os.path.isfile(dataset_path + "/" + img)):
            missing_images.append(dataset_path + "/" + img);
    return missing_images;


@accepts(str, verbose=[bool, str, int], post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def populate_corrupt_from_foldered(dataset_path, verbose=1):
    '''
    Find number of corrupted images in imageset - foldered type

    Args:
        dataset_path (str): Path to dataset containing images
        verbose (str): If True, prints logs for analysis 

    Returns:
        list: List of corrupt images
    '''
    classes_folder = os.listdir(dataset_path);
    corrupt_images = [];
    if(verbose):
        for i in tqdm(range(len(classes_folder))):
            list_imgs = os.listdir(dataset_path + "/" + classes_folder[i]);  
            for j in range(len(list_imgs)):
                img_name = dataset_path + "/" + classes_folder[i] + "/" + list_imgs[j]
                if(os.path.isfile(img_name)):
                    try:
                        img = Image.open(img_name)
                        img.verify()
                    except Exception:
                        corrupt_images.append(img_name)
    else:
        for i in range(len(classes_folder)):
            list_imgs = os.listdir(dataset_path + "/" + classes_folder[i]);  
            for j in range(len(list_imgs)):
                img_name = dataset_path + "/" + classes_folder[i] + "/" + list_imgs[j]
                if(os.path.isfile(img_name)):
                    try:
                        img = Image.open(img_name)
                        img.verify()
                    except Exception:
                        corrupt_images.append(img_name)
    return corrupt_images;


@accepts(str, str, str, verbose=[bool, str, int], post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def populate_corrupt_from_csv(tpath, dataset_path, delimiter, verbose=1):
    '''
    Find number of corrupted images in imageset - csv type

    Args:
        dataset_path (str): Path to dataset containing images
        verbose (str): If True, prints logs for analysis 

    Returns:
        list: List of corrupt images
    '''
    lst = read_csv(tpath);
    corrupt_images = [];
    if(verbose):
        for i in tqdm(range(len(lst))):
            img_name, label = lst[i][:len(lst[i])-1].split(delimiter);
            img_name = dataset_path + "/" + img_name
            if(os.path.isfile(img_name)):
                try:
                    img = Image.open(img_name)
                    img.verify()
                except Exception:
                    corrupt_images.append(img_name)
    else:
        for i in range(len(lst)):
            img_name, label = lst[i][:len(lst[i])-1].split(delimiter);
            img_name = dataset_path + "/" + img_name
            if(os.path.isfile(img_name)):
                try:
                    img = Image.open(img_name)
                    img.verify()
                except Exception:
                    corrupt_images.append(img_name)
    return corrupt_images;