from system.eda.imports import *
from system.imports import *


@accepts(str, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def get_img_label(fname, delimiter):
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

@accepts(str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def read_csv(fname):
    f = open(fname);
    lst = f.readlines();
    f.close();
    del lst[0]
    return lst;

@accepts(str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_folder_train(tpath):
    classes_folder = os.listdir(tpath);
    classes_folder_strength = [];
    for i in range(len(classes_folder)):
        classes_folder_strength.append(len(os.listdir(tpath + "/" + classes_folder[i])));
    return classes_folder, classes_folder_strength;


@accepts(str, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_folder_trainval(tpath, vpath):
    classes_folder, classes_folder_strength = populate_from_folder_train(tpath);
    for i in range(len(os.listdir(vpath))):
            classes_folder_strength[i] = classes_folder_strength[i] + len(os.listdir(vpath + "/" + classes_folder[i]));
    return classes_folder, classes_folder_strength;


@accepts(str, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_csv_train(tpath, delimiter):
    img_list, label_list = get_img_label(tpath, delimiter);
    classes_folder = list(np.unique(sorted(label_list)))
    classes_folder_strength = [];
    for i in range(len(classes_folder)):
        classes_folder_strength.append(label_list.count(classes_folder[i]));
    return classes_folder, classes_folder_strength;

@accepts(str, str, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def populate_from_csv_trainval(tpath, vpath, delimiter):
    classes_folder, classes_folder_strength = populate_from_csv_train(tpath, delimiter);
    img_list, label_list = get_img_label(vpath, delimiter);
    for i in range(len(classes_folder)):
        classes_folder_strength[i] += label_list.count(classes_folder[i]);
    return classes_folder, classes_folder_strength;


@accepts(str, str, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def populate_missing(tpath, dataset_path, delimiter):
    lst = read_csv(tpath);
    missing_images = [];
    for i in range(len(lst)):
        img, label = lst[i][:len(lst[i])-1].split(delimiter);
        if(not os.path.isfile(dataset_path + "/" + img)):
            missing_images.append(dataset_path + "/" + img);
    return missing_images;


@accepts(str, verbose=[bool, str, int], post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def populate_corrupt_from_foldered(dataset_path, verbose=1):
    classes_folder = os.listdir(dataset_path);
    corrupt_images = [];
    if(verbose):
        for i in tqdm(range(len(classes_folder))):
            list_imgs = os.listdir(dataset_path + "/" + classes_folder[i]);  
            for j in range(len(list_imgs)):
                img_name = dataset_path + "/" + classes_folder[i] + "/" + list_imgs[j]
                if(os.path.isfile(img_name)):
                    img = Image.open(img_name)
                    try:
                        img.verify()
                    except Exception:
                        corrupt_images.append(img_name)
    else:
        for i in range(len(classes_folder)):
            list_imgs = os.listdir(dataset_path + "/" + classes_folder[i]);  
            for j in range(len(list_imgs)):
                img_name = dataset_path + "/" + classes_folder[i] + "/" + list_imgs[j]
                if(os.path.isfile(img_name)):
                    img = Image.open(img_name)
                    try:
                        img.verify()
                    except Exception:
                        corrupt_images.append(img_name)
    return corrupt_images;


@accepts(str, str, str, verbose=[bool, str, int], post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def populate_corrupt_from_csv(tpath, dataset_path, delimiter, verbose=1):
    lst = read_csv(tpath);
    corrupt_images = [];
    if(verbose):
        for i in tqdm(range(len(lst))):
            img_name, label = lst[i][:len(lst[i])-1].split(delimiter);
            img_name = dataset_path + "/" + img_name
            if(os.path.isfile(img_name)):
                img = Image.open(img_name)
                try:
                    img.verify()
                except Exception:
                    corrupt_images.append(img_name)
    else:
        for i in range(len(lst)):
            img_name, label = lst[i][:len(lst[i])-1].split(delimiter);
            img_name = dataset_path + "/" + img_name
            if(os.path.isfile(img_name)):
                img = Image.open(img_name)
                try:
                    img.verify()
                except Exception:
                    corrupt_images.append(img_name)
    return corrupt_images;